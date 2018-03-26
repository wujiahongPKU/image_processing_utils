# encoding=utf-8

from __future__ import print_function

from PIL import Image
from grpc.beta import implementations
import numpy
import tensorflow as tf
import cv2

from protos import predict_pb2
from protos import prediction_service_pb2




'''
parse the results and get the final result
'''
def decode_result(result):
    """
    :param result:
    :return:
    """
    d_result = {}
    for key in result.outputs:
        if result.outputs[key].dtype == 1:
            vals = numpy.asarray(result.outputs[key].float_val, dtype=numpy.float32)
        elif result.outputs[key].dtype == 2:
            vals = numpy.asarray(result.outputs[key].double_val, dtype=numpy.double)
        elif result.outputs[key].dtype == 3:
            vals = result.outputs[key].int_val
        elif result.outputs[key].dtype == 4:
            vals = result.outputs[key].uint8_val
        elif result.outputs[key].dtype == 5:
            vals = result.outputs[key].int16_val
        elif result.outputs[key].dtype == 6:
            vals = result.outputs[key].int8_val
        elif result.outputs[key].dtype == 7:
            vals = result.outputs[key].string_val
        elif result.outputs[key].dtype == 8:
            vals = result.outputs[key].complex64_val
        elif result.outputs[key].dtype == 9:
            vals = numpy.asarray(result.outputs[key].int64_val, dtype=numpy.int64)
        elif result.outputs[key].dtype == 10:
            vals = result.outputs[key].bool_val
        else:
            raise Exception("Unknown dtype.")
        shp = [dim.size for dim in result.outputs[key].tensor_shape.dim]
        d_result[key] = numpy.reshape(vals, shp)
    '''
    d_result is a dict the key is like :
    "classes_num":
    ""
    '''
    return d_result


class image_Detector(object):
    def __init__(self, host, port):
        self.channel = implementations.insecure_channel(host, int(port))
        self.stub = prediction_service_pb2.beta_create_PredictionService_stub(self.channel)

    def detection_only(self, image, callback=None, timeout=30):
        """ detection only,
        :param image: Image.Image object
        :param callback: callback function
        :param timeout:
        :return:
        if callback is None, return a result dict {'scores': Arr[1*N], 'classes': Arr[N], 'boxes': Arr[1*N*4]},
        if define callback function, this function will process the upper result.
        """
        assert isinstance(image, Image.Image)

        request = predict_pb2.PredictRequest()
        request.model_spec.name = 'fridge_server'
        request.model_spec.signature_name = 'detection_only'
        ratio = max(image.size) / 1024
        if ratio > 1:
            image = image.resize([int(round(x / ratio)) for x in image.size])
        exp_image_arr = numpy.expand_dims(numpy.asarray(image), axis=0)
        request.inputs['inputs'].CopyFrom(
            tf.contrib.util.make_tensor_proto(exp_image_arr, shape=exp_image_arr.shape))

        def _callback(res):
            d_res = decode_result(res.result())
            return callback(d_res)

        if callback is not None:
            result_future = self.stub.Predict.future(request, timeout)
            result_future.add_done_callback(_callback)
        else:
            result_future = self.stub.Predict(request, timeout)
            result_future = decode_result(result_future)
            return result_future

    def classify_only(self, image, callback=None, timeout=30):
        """ classify only
        :param image: Image.Image object
        :param callback: callback function
        :param timeout:
        :return:
        if callback is None, return a result dict {'scores': Arr[1*1], 'classes': Arr[1]},
        if define callback function, this function will process the upper result.
        """
        assert isinstance(image, Image.Image)

        request = predict_pb2.PredictRequest()
        request.model_spec.name = 'mnist'
        request.model_spec.signature_name = 'classify_only'
        ratio = max(image.size) / 128
        if ratio > 1:
            image = image.resize([int(round(x / ratio)) for x in image.size])
        exp_image_arr = numpy.expand_dims(numpy.asarray(image), axis=0)
        request.inputs['inputs'].CopyFrom(
            tf.contrib.util.make_tensor_proto(exp_image_arr, shape=exp_image_arr.shape))

        def _callback(res):
            d_res = decode_result(res.result())
            return callback(d_res)

        if callback is not None:
            result_future = self.stub.Predict.future(request, timeout)
            result_future.add_done_callback(_callback)
        else:
            result_future = self.stub.Predict(request, timeout)
            result_future = decode_result(result_future)
            return result_future

    def detection_classify(self, image, callback=None, timeout=30):
        """ detection only,
        :param image: Image.Image object
        :param callback: callback function
        :param timeout:
        :return:
        if callback is None, return a result dict {'scores': Arr[N], 'classes': Arr[N], 'boxes': Arr[1*N*4]},
        if define callback function, this function will process the upper result.
        """
        assert isinstance(image, Image.Image)

        request = predict_pb2.PredictRequest()
        request.model_spec.name = 'mnist'
        request.model_spec.signature_name = 'detection_classify'
        ratio = max(image.size) / 1024
        if ratio > 1:
            image = image.resize([int(round(x / ratio)) for x in image.size])
        exp_image_arr = numpy.expand_dims(numpy.asarray(image), axis=0)
        request.inputs['inputs'].CopyFrom(
            tf.contrib.util.make_tensor_proto(exp_image_arr, shape=exp_image_arr.shape))

        def _callback(res):
            d_res = decode_result(res.result())
            return callback(d_res)

        if callback is not None:
            result_future = self.stub.Predict.future(request, timeout)
            result_future.add_done_callback(_callback)
        else:
            result_future = self.stub.Predict(request, timeout)
            result_future = decode_result(result_future)
            return result_future

    def get_detection_only_class2index_map(self, callback=None, timeout=30):
        request = predict_pb2.PredictRequest()
        request.model_spec.name = 'mnist'
        request.model_spec.signature_name = 'detection_only_class2index_map'
        fake_input = numpy.asarray(1, dtype=numpy.int32)
        request.inputs['inputs'].CopyFrom(
            tf.contrib.util.make_tensor_proto(fake_input, shape=fake_input.shape))

        def _callback(res):
            d_res = decode_result(res.result())
            return callback(d_res)

        if callback is not None:
            result_future = self.stub.Predict.future(request, timeout)
            result_future.add_done_callback(_callback)
        else:
            result_future = self.stub.Predict(request, timeout)
            result_future = decode_result(result_future)
            return result_future

    def get_classify_only_class2index_map(self, callback=None, timeout=30):
        request = predict_pb2.PredictRequest()
        request.model_spec.name = 'mnist'
        request.model_spec.signature_name = 'classify_only_class2index_map'
        fake_input = numpy.asarray(1, dtype=numpy.int32)
        request.inputs['inputs'].CopyFrom(
            tf.contrib.util.make_tensor_proto(fake_input, shape=fake_input.shape))

        def _callback(res):
            d_res = decode_result(res.result())
            return callback(d_res)

        if callback is not None:
            result_future = self.stub.Predict.future(request, timeout)
            result_future.add_done_callback(_callback)
        else:
            result_future = self.stub.Predict(request, timeout)
            result_future = decode_result(result_future)
            return result_future

    def get_detection_classify_class2index_map(self, callback=None, timeout=30):
        request = predict_pb2.PredictRequest()
        request.model_spec.name = 'mnist'
        request.model_spec.signature_name = 'detection_classify_class2index_map'
        fake_input = numpy.asarray(1, dtype=numpy.int32)
        request.inputs['inputs'].CopyFrom(
            tf.contrib.util.make_tensor_proto(fake_input, shape=fake_input.shape))

        def _callback(res):
            d_res = decode_result(res.result())
            return callback(d_res)

        if callback is not None:
            result_future = self.stub.Predict.future(request, timeout)
            result_future.add_done_callback(_callback)
        else:
            result_future = self.stub.Predict(request, timeout)
            result_future = decode_result(result_future)
            return result_future



if __name__=="__main__":
    host="10.18.103.205"
    port=9888
    test_image_path="./image_test/test_1_0.png"
    image_detection=image_Detector(host,port)
    image_data=Image.open(test_image_path)
    result_feature=image_detection.detection_only(image_data)
    print ("the result feature is:",result_feature)
    # print("test this part and get final results")