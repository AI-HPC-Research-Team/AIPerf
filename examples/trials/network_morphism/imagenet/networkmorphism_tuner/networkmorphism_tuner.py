# Copyright (c) Microsoft Corporation.
# Copyright (c) Peng Cheng Laboratory.
# Copyright (c) Tsinghua University.
# Licensed under the MIT license.

"""
networkmorphsim_tuner.py
"""

import logging
import os

from nni.tuner import Tuner
from nni.utils import OptimizeMode, extract_scalar_reward
from nni.networkmorphism_tuner.bayesian import BayesianOptimizer
from nni.networkmorphism_tuner.nn import CnnGenerator, MlpGenerator
from nni.networkmorphism_tuner.utils import Constant
from nni.networkmorphism_tuner.graph import graph_to_json, json_to_graph

# v1.1
import multiprocessing
import time
import nni
logger = logging.getLogger("NetworkMorphism_AutoML")
lock=multiprocessing.Lock()

class NetworkMorphismTuner(Tuner):
    """
    NetworkMorphismTuner is a tuner which using network morphism techniques.

    Attributes
    ----------
    n_classes : int
        The class number or output node number (default: ``10``)
    input_shape : tuple
        A tuple including: (input_width, input_width, input_channel)
    t_min : float
        The minimum temperature for simulated annealing. (default: ``Constant.T_MIN``)
    beta : float
        The beta in acquisition function. (default: ``Constant.BETA``)
    algorithm_name : str
        algorithm name used in the network morphism (default: ``"Bayesian"``)
    optimize_mode : str
        optimize mode "minimize" or "maximize" (default: ``"minimize"``)
    verbose : bool
        verbose to print the log (default: ``True``)
    bo : BayesianOptimizer
        The optimizer used in networkmorphsim tuner.
    max_model_size : int
        max model size to the graph (default: ``Constant.MAX_MODEL_SIZE``)
    default_model_len : int
        default model length (default: ``Constant.MODEL_LEN``)
    default_model_width : int
        default model width (default: ``Constant.MODEL_WIDTH``)
    search_space : dict
    """

    def __init__(
            self,
            task="cv",
            input_width=32,
            input_channel=3,
            n_output_node=10,
            algorithm_name="Bayesian",
            optimize_mode="maximize",
            path="model_path",
            verbose=True,
            beta=Constant.BETA,
            t_min=Constant.T_MIN,
            max_model_size=Constant.MAX_MODEL_SIZE,
            default_model_len=Constant.MODEL_LEN,
            default_model_width=Constant.MODEL_WIDTH,
    ):
        """
        initilizer of the NetworkMorphismTuner.
        """
        # v1.1
        exp_id = os.getcwd().split("/")[-2]
        logPath = os.environ["HOME"] + "/mountdir/nni/experiments/" + exp_id + "/log"
        self.path = os.path.join(logPath, path)
        logger.info("self.path = " + self.path)

        if not os.path.exists(self.path):
            os.makedirs(self.path)
        if task == "cv":
            self.generators = [CnnGenerator]
        elif task == "common":
            self.generators = [MlpGenerator]
        else:
            raise NotImplementedError(
                '{} task not supported in List ["cv","common"]')

        self.n_classes = n_output_node
        self.input_shape = (input_width, input_width, input_channel)

        self.t_min = t_min
        self.beta = beta
        self.algorithm_name = algorithm_name
        self.optimize_mode = OptimizeMode(optimize_mode)
        self.json = None
        self.total_data = {}
        self.verbose = verbose
        self.model_count = 0

        self.bo = BayesianOptimizer(
            self, self.t_min, self.optimize_mode, self.beta)
        self.training_queue = []
        self.descriptors = []
        self.history = []

        self.max_model_size = max_model_size
        self.default_model_len = default_model_len
        self.default_model_width = default_model_width

        self.search_space = dict()


    def update_search_space(self, search_space):
        """
        Update search space definition in tuner by search_space in neural architecture.
        """
        self.search_space = search_space
    
    # v1.1
    def set_descriptors(self, model_id, generated_graph):
        self.descriptors[model_id] = generated_graph.extract_descriptor()

    def fake_generate_parameters(self, parameter_id, **kwargs):
        """
        Returns a initialized model.
        """
        self.init_search()

        new_father_id = None
        generated_graph = None

        graph, father_id, model_id = self.training_queue.pop(0)

        # from graph to json
        json_model_path = os.path.join(self.path, str(model_id) + ".json")
        json_out = graph_to_json(graph, json_model_path)
        self.total_data[parameter_id] = (json_out, father_id, model_id)

        return json_out

    def generate_parameters(self, parameter_id, **kwargs):
        """
        Returns a set of trial neural architecture, as a serializable object.

        Parameters
        ----------
        parameter_id : int
        """
        '''
        if not self.history:
            self.init_search()

        new_father_id = None
        generated_graph = None
        if not self.training_queue:
            new_father_id, generated_graph = self.generate()
            new_model_id = self.model_count
            self.model_count += 1
            self.training_queue.append(
                (generated_graph, new_father_id, new_model_id))
            self.descriptors.append(generated_graph.extract_descriptor())

        graph, father_id, model_id = self.training_queue.pop(0)

        # from graph to json
        json_model_path = os.path.join(self.path, str(model_id) + ".json")
        json_out = graph_to_json(graph, json_model_path)
        self.total_data[parameter_id] = (json_out, father_id, model_id)

        return json_out
        '''
        if not self.history:
            #self.init_search()
            print("If there is no history, generate_parameters should not be called!")
            exit(1)
        total_start=time.time()

        rate = 2
        
        f11=open("/root/xlei","a+")
        f11.write("tuner.generate:"+"total time:"+str(total_start)+"\n")
        f11.close()

        if (os.path.exists(os.environ["HOME"] + "/mountdir/nni/experiments/" + str(nni.get_experiment_id()) + "/g_time") and os.path.exists(os.environ["HOME"] + "/mountdir/nni/experiments/" + str(nni.get_experiment_id()) + "/c_time")):
            f3 = open(os.environ["HOME"] + "/mountdir/nni/experiments/" + str(nni.get_experiment_id()) + "/g_time", "r")
            g_t = float(f3.read())
            f4 = open(os.environ["HOME"] + "/mountdir/nni/experiments/" + str(nni.get_experiment_id()) + "/c_time", "r")
            c_t = float(f4.read())
            f11=open("/root/xlei","a+")
            f11.write("tuner.generate:"+"total time:"+str(total_start)+"\n")
            f11.close()
            if (g_t != 0) and (c_t != 0):
                realrate = int(c_t / g_t)
                # f5 = open("/root/rate", "a+")
                # f5.write(str(realrate))
                # f5.write(" ")
                # if realrate < 0:
                #     realrate = 0 - realrate
                if (realrate < 5) and (realrate > 1):
                    rate = int(realrate)
                elif (realrate <= 1):
                    rate = 1

        for i in range(rate):
            start=time.time()
            new_father_id = None
            generated_graph = None
            f11=open("/root/xlei","a+")
            f11.write("tuner.generate:"+"total time:"+str(total_start)+"\n")
            f11.close()
            if not self.training_queue:
                f11=open("/root/xlei","a+")
                f11.write("self.training_queue: "+ str(self.training_queue)+'\n')
                f11.close()
                new_father_id, generated_graph = self.generate()
                father_id,json_out,new_model_id = self.total_data[parameter_id]
                self.training_queue.append((generated_graph, new_father_id, new_model_id))
                #self.descriptors.append(generated_graph.extract_descriptor())
            else:
                f11=open("/root/xlei","a+")
                f11.write("tuner.generate: self.training_queue")
                f11.close()
                print("training_queue should be an empty list.")
                exit(1)

            graph, father_id, model_id = self.training_queue.pop(0)
        # from graph to json
            json_model_path = os.path.join(self.path, str(model_id) + ".json")
            f11=open('/root/json','a+')
            f11.write(str(json_model_path)+'\n')
            f11.close()
            json_out = graph_to_json(graph, json_model_path)
            end=time.time()
        #self.total_data[parameter_id] = (json_out, father_id, model_id)
            json_and_id="json_out="+str(json_out)+"+father_id="+str(father_id)+"+parameter_id="+str(parameter_id)+"+history="+"True"
            lock.acquire()
            f1=open(os.environ["HOME"] + "/mountdir/nni/experiments/" + str(nni.get_experiment_id()) + "/trials/" + str(nni.get_trial_id()) + "/output.log","a+")
            f1.write("single_generate=" + str(end - start)+"\n")
            f1.close()
            f=open(os.environ["HOME"] + "/mountdir/nni/experiments/" + str(nni.get_experiment_id()) + "/graph.txt","a+")
            f.write(json_and_id+"\n")
            f.close()
            lock.release()
        total_end=time.time()
        lock.acquire()
        f1=open(os.environ["HOME"] + "/mountdir/nni/experiments/" + str(nni.get_experiment_id()) + "/trials/" + str(nni.get_trial_id()) + "/output.log","a+")
        f1.write("total_generate=" + str(total_end - total_start)+"\n")
        f1.close()
        lock.release()

        totime = total_end - total_start
        if totime<0:
            totime = 0-totime

        f1 = open (os.environ["HOME"] + "/mountdir/nni/experiments/" + str(nni.get_experiment_id()) + "/g_time","w+")
        gt = totime/rate
        f1.write(str(gt))
        f1.close()

    def receive_trial_result(self, parameter_id, parameters, value, **kwargs):
        """
        Record an observation of the objective function.

        Parameters
        ----------
        parameter_id : int
            the id of a group of paramters that generated by nni manager.
        parameters : dict
            A group of parameters.
        value : dict/float
            if value is dict, it should have "default" key.
        """
        reward = extract_scalar_reward(value)

        if parameter_id not in self.total_data:
            raise RuntimeError("Received parameter_id not in total_data.")

        (_, father_id, model_id) = self.total_data[parameter_id]

        graph = self.bo.searcher.load_model_by_id(model_id)

        # to use the value and graph
        self.add_model(reward, model_id)
        self.update(father_id, graph, reward, model_id)


    def init_search(self):
        """
        Call the generators to generate the initial architectures for the search.
        """
        # v1.1
        #if self.verbose:
        #    logger.info("Initializing search.")
        for generator in self.generators:
            graph = generator(self.n_classes, self.input_shape).generate(
                self.default_model_len, self.default_model_width
            )
            model_id = self.model_count
            self.model_count += 1
            self.training_queue.append((graph, -1, model_id))
            self.descriptors.append(graph.extract_descriptor())

        #if self.verbose:
        #    logger.info("Initialization finished.")


    def generate(self):
        """
        Generate the next neural architecture.

        Returns
        -------
        other_info : any object
            Anything to be saved in the training queue together with the architecture.
        generated_graph : Graph
            An instance of Graph.
        """
        generated_graph, new_father_id = self.bo.generate(self.descriptors)
        if new_father_id is None:
            new_father_id = 0
            generated_graph = self.generators[0](
                self.n_classes, self.input_shape
            ).generate(self.default_model_len, self.default_model_width)

        return new_father_id, generated_graph

    def update(self, other_info, graph, metric_value, model_id):
        """
        Update the controller with evaluation result of a neural architecture.

        Parameters
        ----------
        other_info: any object
            In our case it is the father ID in the search tree.
        graph: Graph
            An instance of Graph. The trained neural architecture.
        metric_value: float
            The final evaluated metric value.
        model_id: int
        """
        '''
        father_id = other_info
        self.bo.fit([graph.extract_descriptor()], [metric_value])
        self.bo.add_child(father_id, model_id)
        '''
        # v1.1
        father_id = other_info
        t1 = time.time()
        self.bo.fit([graph.extract_descriptor()], [metric_value])
        self.bo.add_child(father_id, model_id)
        t2 = time.time()
        print("Update time = " + str(t2 - t1))
        
    def add_model(self, metric_value, model_id):
        """
        Add model to the history, x_queue and y_queue

        Parameters
        ----------
        metric_value : float
        graph : dict
        model_id : int

        Returns
        -------
        model : dict
        """
        if self.verbose:
            logger.info("Saving model.")

        # Update best_model text file
        ret = {"model_id": model_id, "metric_value": metric_value}
        self.history.append(ret)
        if model_id == self.get_best_model_id():
            file = open(os.path.join(self.path, "best_model.txt"), "w")
            file.write("best model: " + str(model_id))
            file.close()
        return ret


    def get_best_model_id(self):
        """
        Get the best model_id from history using the metric value
        """

        if self.optimize_mode is OptimizeMode.Maximize:
            return max(self.history, key=lambda x: x["metric_value"])[
                "model_id"]
        return min(self.history, key=lambda x: x["metric_value"])["model_id"]


    def load_model_by_id(self, model_id):
        """
        Get the model by model_id

        Parameters
        ----------
        model_id : int
            model index

        Returns
        -------
        load_model : Graph
            the model graph representation
        """

        with open(os.path.join(self.path, str(model_id) + ".json")) as fin:
            json_str = fin.read().replace("\n", "")

        load_model = json_to_graph(json_str)
        return load_model

    def load_best_model(self):
        """
        Get the best model by model id

        Returns
        -------
        load_model : Graph
            the model graph representation
        """
        return self.load_model_by_id(self.get_best_model_id())

    def get_metric_value_by_id(self, model_id):
        """
        Get the model metric valud by its model_id

        Parameters
        ----------
        model_id : int
            model index

        Returns
        -------
        float
             the model metric
        """
        for item in self.history:
            if item["model_id"] == model_id:
                return item["metric_value"]
        return None

    def import_data(self, data):
        pass

