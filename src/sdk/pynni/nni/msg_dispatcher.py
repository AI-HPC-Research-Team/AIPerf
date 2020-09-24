# Copyright (c) Microsoft Corporation.
# Copyright (c) Tsinghua University.
# Copyright (c) Peng Cheng Laboratory.
# Licensed under the MIT license.

import logging
from collections import defaultdict
import json_tricks

from nni import NoMoreTrialError
from .protocol import CommandType, send
from .msg_dispatcher_base import MsgDispatcherBase
from .assessor import AssessResult
from .common import multi_thread_enabled, multi_phase_enabled
from .env_vars import dispatcher_env_vars
from .utils import MetricType, to_json

# add from v1.1 -
import zmq
import threading as th
import sys
from nni.networkmorphism_tuner.graph import json_to_graph
# --

_logger = logging.getLogger(__name__)
lock = th.Lock()

# Assessor global variables
_trial_history = defaultdict(dict)
'''key: trial job ID; value: intermediate results, mapping from sequence number to data'''

_ended_trials = set()
'''trial_job_id of all ended trials.
We need this because NNI manager may send metrics after reporting a trial ended.
TODO: move this logic to NNI manager
'''


def _sort_history(history):
    ret = []
    for i, _ in enumerate(history):
        if i in history:
            ret.append(history[i])
        else:
            break
    return ret


# Tuner global variables
_next_parameter_id = 0
_trial_params = {}
'''key: trial job ID; value: parameters'''
_customized_parameter_ids = set()


def _create_parameter_id():
    global _next_parameter_id
    _next_parameter_id += 1
    return _next_parameter_id - 1


def _pack_parameter(parameter_id, params, customized=False, trial_job_id=None, parameter_index=None):
    lock.acquire()
    _trial_params[parameter_id] = params
    lock.release()
    ret = {
        'parameter_id': parameter_id,
        'parameter_source': 'customized' if customized else 'algorithm',
        'parameters': params
    }
    if trial_job_id is not None:
        ret['trial_job_id'] = trial_job_id
    if parameter_index is not None:
        ret['parameter_index'] = parameter_index
    else:
        ret['parameter_index'] = 0
    return to_json(ret)
    #return json_tricks.dumps(ret) # v1.1 -------------------------------------


class MsgDispatcher(MsgDispatcherBase):
    def __init__(self, tuner, assessor=None):
        super(MsgDispatcher, self).__init__()
        self.tuner = tuner
        self.assessor = assessor
        if assessor is None:
            _logger.debug('Assessor is not configured')
        self.current_jobs = 0
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind("tcp://0.0.0.0:800081")
        self.zmq_server_thread = th.Thread(target = MsgDispatcher.zmq_server_func, args=(self,))
        self.zmq_server_thread.setDaemon(True)
        self.zmq_server_thread.start()

    def zmq_server_func(self):
        while True:
            try:
                message = self.socket.recv_pyobj()
                if message["type"] == "get_next_parameter":
                    ret = {}
                    lock.acquire()
                    ret["tuner"] = self.tuner

                    self.socket.send_pyobj(ret)
                    lock.release()
                elif message["type"] == "generated_parameter":
                    self.current_jobs += 1
                    print("New model generated, current jobs = " + str(self.current_jobs))
                    if not "parameters" in message:
                        self.socket.send_pyobj("nothing")
                        continue
                    parameter_id = message["parameter_id"]
                    father_id = message["father_id"]
                    json_params = message["parameters"]
                    lock.acquire()
                    x,y,model_id = self.tuner.total_data[parameter_id]
                    generated_graph = json_to_graph(json_params)
                    self.tuner.set_descriptors(model_id, generated_graph)
                    self.tuner.total_data[parameter_id] = (json_params, father_id, model_id)
                    _trial_params[parameter_id] = json_params
                    lock.release()
                    self.socket.send_pyobj("nothing")
                elif message["type"] == "FINAL":
                    self.socket.send_pyobj("nothing")
                    self._handle_final_metric_data(message)
            except Exception as e:
                print('error:',e)
                sys.exit()

    def _on_exit(self):
        print("dispatcher _on_exit")
        #self.zmq_server_thread.join()
        self.socket.unbind("tcp://0.0.0.0:800081")

    def load_checkpoint(self):
        self.tuner.load_checkpoint()
        if self.assessor is not None:
            self.assessor.load_checkpoint()

    def save_checkpoint(self):
        self.tuner.save_checkpoint()
        if self.assessor is not None:
            self.assessor.save_checkpoint()

    def handle_initialize(self, data):
        """Data is search space
        """
        lock.acquire()
        self.tuner.update_search_space(data)
        lock.release()
        send(CommandType.Initialized, '')

    def send_trial_callback(self, id_, params):
        """For tuner to issue trial config when the config is generated
        """
        send(CommandType.NewTrialJob, _pack_parameter(id_, params))

    def handle_request_trial_jobs(self, data):
        # data: number or trial jobs
        ids = [_create_parameter_id() for _ in range(data)]
        _logger.debug("requesting for generating params of %s", ids)
        lock.acquire()
        #params_list = self.tuner.generate_multiple_parameters(ids, st_callback=self.send_trial_callback)
        params_list = self.tuner.fake_generate_multiple_parameters(ids) #v1.1 ------
        lock.release()
        for i, _ in enumerate(params_list):
            send(CommandType.NewTrialJob, _pack_parameter(ids[i], params_list[i]))
        # when parameters is None.
        if len(params_list) < len(ids):
            send(CommandType.NoMoreTrialJobs, _pack_parameter(ids[0], ''))

    def handle_update_search_space(self, data):
        lock.acquire()
        self.tuner.update_search_space(data)
        lock.release()

    def handle_import_data(self, data):
        """Import additional data for tuning
        data: a list of dictionaries, each of which has at least two keys, 'parameter' and 'value'
        """
        #for entry in data:
        #    entry['value'] = json_tricks.loads(entry['value'])
        #self.tuner.import_data(data)
        
        # v1.1
        lock.acquire()
        self.tuner.import_data(data)
        lock.release()

    def handle_add_customized_trial(self, data):
        # data: parameters
        id_ = _create_parameter_id()
        _customized_parameter_ids.add(id_)
        send(CommandType.NewTrialJob, _pack_parameter(id_, data, customized=True)) # v1.1

    def handle_report_metric_data(self, data):
        """
        data: a dict received from nni_manager, which contains:
              - 'parameter_id': id of the trial
              - 'value': metric value reported by nni.report_final_result()
              - 'type': report type, support {'FINAL', 'PERIODICAL'}
        """
        # metrics value is dumped as json string in trial, so we need to decode it here
        #if 'value' in data:
        #    data['value'] = json_tricks.loads(data['value'])
        if data['type'] == MetricType.FINAL:
            self._handle_final_metric_data(data)
        elif data['type'] == MetricType.PERIODICAL:
            if self.assessor is not None:
                self._handle_intermediate_metric_data(data)
        elif data['type'] == MetricType.REQUEST_PARAMETER:
            print("REQUEST_PARAMETER is not supported.")
            exit(1)  # v1.1
            assert multi_phase_enabled()
            assert data['trial_job_id'] is not None
            assert data['parameter_index'] is not None
            param_id = _create_parameter_id()
            try:
                lock.acquire()
                param = self.tuner.generate_parameters(param_id, trial_job_id=data['trial_job_id'])
                lock.release()
            except NoMoreTrialError:
                param = None
            send(CommandType.SendTrialJobParameter, _pack_parameter(param_id, param, trial_job_id=data['trial_job_id'],
                                                                    parameter_index=data['parameter_index']))
        else:
            raise ValueError('Data type not supported: {}'.format(data['type']))

    def handle_trial_end(self, data):
        """
        data: it has three keys: trial_job_id, event, hyper_params
             - trial_job_id: the id generated by training service
             - event: the job's state
             - hyper_params: the hyperparameters generated and returned by tuner
        """
        trial_job_id = data['trial_job_id']
        _ended_trials.add(trial_job_id)
        if trial_job_id in _trial_history:
            _trial_history.pop(trial_job_id)
            if self.assessor is not None:
                self.assessor.trial_end(trial_job_id, data['event'] == 'SUCCEEDED')
        lock.acquire()
        if self.tuner is not None:
            self.tuner.trial_end(json_tricks.loads(data['hyper_params'])['parameter_id'], data['event'] == 'SUCCEEDED')
        lock.release()

    def _handle_final_metric_data(self, data):
        """Call tuner to process final results
        """
        id_ = data['parameter_id']
        value = data['value']
        lock.acquire()
        if id_ is None or id_ in _customized_parameter_ids:
            '''
            if not hasattr(self.tuner, '_accept_customized'):
                self.tuner._accept_customized = False
            if not self.tuner._accept_customized:
                _logger.info('Customized trial job %s ignored by tuner', id_)
                return
            '''
            # v1.1
            if multi_phase_enabled():
                self.tuner.receive_customized_trial_result(id_, _trial_params[id_], value, trial_job_id=data['trial_job_id'])
            else:
                self.tuner.receive_customized_trial_result(id_, _trial_params[id_], value)
            customized = True
        else:
            customized = False
            #self.tuner.receive_trial_result(id_, _trial_params[id_], value, customized=customized,
            #                                trial_job_id=data.get('trial_job_id'))
            # v1.1
            if multi_phase_enabled():
                self.tuner.receive_trial_result(id_, _trial_params[id_], value, trial_job_id=data['trial_job_id'])
            else:
                self.tuner.receive_trial_result(id_, _trial_params[id_], value)
        lock.release()

    def _handle_intermediate_metric_data(self, data):
        """Call assessor to process intermediate results
        """
        if data['type'] != MetricType.PERIODICAL:
            return
        if self.assessor is None:
            return

        trial_job_id = data['trial_job_id']
        if trial_job_id in _ended_trials:
            return

        history = _trial_history[trial_job_id]
        history[data['sequence']] = data['value']
        ordered_history = _sort_history(history)
        if len(ordered_history) < data['sequence']:  # no user-visible update since last time
            return

        try:
            result = self.assessor.assess_trial(trial_job_id, ordered_history)
        except Exception as e:
            _logger.error('Assessor error')
            _logger.exception(e)

        if isinstance(result, bool):
            result = AssessResult.Good if result else AssessResult.Bad
        elif not isinstance(result, AssessResult):
            msg = 'Result of Assessor.assess_trial must be an object of AssessResult, not %s'
            raise RuntimeError(msg % type(result))

        if result is AssessResult.Bad:
            _logger.debug('BAD, kill %s', trial_job_id)
            send(CommandType.KillTrialJob, json_tricks.dumps(trial_job_id))
            # notify tuner
            _logger.debug('env var: NNI_INCLUDE_INTERMEDIATE_RESULTS: [%s]',
                          dispatcher_env_vars.NNI_INCLUDE_INTERMEDIATE_RESULTS)
            if dispatcher_env_vars.NNI_INCLUDE_INTERMEDIATE_RESULTS == 'true':
                self._earlystop_notify_tuner(data)
        else:
            _logger.debug('GOOD')

    def _earlystop_notify_tuner(self, data):
        """Send last intermediate result as final result to tuner in case the
        trial is early stopped.
        """
        _logger.debug('Early stop notify tuner data: [%s]', data)
        data['type'] = MetricType.FINAL
        if multi_thread_enabled():
            self._handle_final_metric_data(data)
        else:
            data['value'] = to_json(data['value'])
            self.enqueue_command(CommandType.ReportMetricData, data)
