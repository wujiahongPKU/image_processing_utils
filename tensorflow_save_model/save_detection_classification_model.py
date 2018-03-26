#coding=utf-8

import os
import sys
import re
import json


import numpy as np
import tensorflow as tf


build_tensor_info = tf.saved_model.utils.build_tensor_info

FLAG = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("detection_model_path", "",
                           "detection")
tf.app.flags.DEFINE_string("classify_model_path", "",
                           "classify")
tf.app.flags.DEFINE_string("detection_label_index_map_path", "",
                           "detection")
tf.app.flags.DEFINE_string("classify_label_index_map_path", "",
                           "classify")
tf.app.flags.DEFINE_string("export_dir", "",
                           "Exported directory")
tf.app.flags.DEFINE_integer("version", 0,
                            "Set exported model version. If don't set, version will increment.")

def load_graph(graph_def_path, input_map=None):
    graph = tf.get_default_graph()
    with graph.as_default():
        graph_def = tf.GraphDef()
        with tf.gfile.GFile(graph_def_path, "rb") as fid:
            serialized_graph = fid.read()
            graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(graph_def, name='', input_map=input_map)
    return graph


def load_frozen_detection_graph(graph_def_path, input_map):
    detection_graph = load_graph(graph_def_path, input_map=input_map)
    # get_detection_graph_inputs_outputs
    input_images = detection_graph.get_tensor_by_name("image_tensor:0")
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    outputs = {"boxes": detection_boxes, "scores": detection_scores,
               "classes": detection_classes, "num_detections": num_detections}

    return detection_graph, input_images, outputs


def load_frozen_classify_graph(graph_def_path, input_map):
    classify_graph = load_graph(graph_def_path, input_map=input_map)
    # get_classify_graph_inputs_outputs
    input_images = classify_graph.get_tensor_by_name("images:0")
    # predict = classify_graph.get_tensor_by_name("predict:0")
    # logits = classify_graph.get_tensor_by_name("logits:0")
    logits=classify_graph.get_tensor_by_name('MobileNet/Predictions/Softmax:0')
    outputs = {"logits": logits}


    return classify_graph, input_images, outputs





def build_merged_detection_classification(detection_graph_def_path, classify_graph_def_path,
                                          detection_class_index_map,
                                          classification_class_index_map,
                                          detection_score_threshold=0.5):
    graph = tf.Graph()
    with graph.as_default():
        encoded_image = tf.placeholder(tf.string, name="detection_encoded_image")
        image_tensor = tf.image.decode_image(encoded_image, channels=3)
        shp = tf.shape(image_tensor)
        input_image_tensor = tf.reshape(image_tensor, tf.stack([-1, shp[-3], shp[-2], 3]), "detection_image_tensor")
        # detection phase
        detection_graph, d_input_images, d_outputs = \
            load_frozen_detection_graph(detection_graph_def_path,
                                        input_map={"image_tensor:0": input_image_tensor})

        # drop boxes which's score is under a given threshold
        valid = tf.greater_equal(tf.reshape(d_outputs["scores"], [-1]),
                                 detection_score_threshold)
        # num = tf.argmin(tf.cast(valid, tf.float32), output_type=tf.int32)
        num = tf.reduce_sum(tf.cast(valid, tf.int32))
        d_boxes = tf.slice(d_outputs["boxes"], tf.stack([0, 0, 0]), tf.stack([-1, num, -1]),
                           name="detection_only_boxes")
        d_scores = tf.slice(d_outputs["scores"], tf.stack([0, 0]), tf.stack([-1, num]))
        d_scores = tf.squeeze(d_scores, axis=0, name="detection_only_scores")
        d_classes = tf.slice(d_outputs["classes"], tf.stack([0, 0]), tf.stack([-1, num]))
        d_classes = tf.squeeze(d_classes, axis=0, name="detection_only_classes")

        ind = tf.reshape(tf.range(tf.shape(d_boxes)[0]), [-1, 1])
        box_ind = ind * tf.ones(tf.shape(d_boxes)[0:2], dtype=tf.int32)
        detect_images = tf.image.crop_and_resize(input_image_tensor,
                                                 boxes=tf.reshape(d_boxes, [-1, 4]),
                                                 box_ind=tf.reshape(box_ind, [-1]),
                                                 crop_size=tf.constant([128, 128]))
        detect_images = tf.cast(detect_images, tf.float32)
        shp = tf.shape(detect_images)
        patch_images = tf.reshape(detect_images, tf.stack((shp[0], shp[1], shp[2], 3)),
                                  name="classify_image_tensor")
        # classification phase
        classify_graph, c_input_images, c_outputs = \
            load_frozen_classify_graph(classify_graph_def_path,
                                       input_map={"images": patch_images})
        c_outputs["scores"] = tf.reduce_max(tf.nn.softmax(c_outputs["logits"]), axis=-1)

        c_scores = tf.identity(c_outputs["scores"], name="classify_scores")
        # c_classes = tf.identity(c_outputs["predict"], name="classify_classes")
        # pred = tf.slice(c_outputs["logits"], tf.stack([0, 0]), tf.stack([num, -1]))
        # scores = tf.slice(c_outputs["scores"], tf.stack([0]), tf.stack([num]),
        #                   name="detection_classify_scores")
        # classes = tf.slice(c_outputs["predict"], tf.stack([0]), tf.stack([num]),
        #                    name="detection_classify_classes")
        scores = tf.identity(c_outputs["scores"], name="detection_classify_scores")
        # classes = tf.identity(c_outputs["predict"], name="detection_classify_classes")

        # add class index map
        fake_input = tf.placeholder(tf.int32)
        det_class2index_map = {key: tf.constant(val) for key, val in detection_class_index_map.items()}
        cls_class2index_map = {key: tf.constant(val) for key, val in classification_class_index_map.items()}

        inputs = {
            "detection_encoded_image": encoded_image,
            "detection_image_tensor": input_image_tensor,
            "classify_image_tensor": patch_images,
            "fake_input": fake_input
        }
        outputs = {
            "classify_scores": c_scores,
            # "classify_classes": c_classes,
            "detection_only_boxes": d_boxes,
            "detection_only_scores": d_scores,
            "detection_only_classes": d_classes,
            "detection_classify_scores": scores,
            # "detection_classify_classes": classes,
            "detection_class_index_map": det_class2index_map,
            "classify_class_index_map": cls_class2index_map
        }

    return graph, inputs, outputs


def save_detection_classify_model(graph, inputs, outputs, saved_model_path):
    builder = tf.saved_model.builder.SavedModelBuilder(saved_model_path)
    with tf.Session(graph=graph) as sess:
        # build tensor info of inputs outputs
        for key in inputs:
            inputs[key] = build_tensor_info(inputs[key])
        for key in outputs:
            if isinstance(outputs[key], tf.Tensor):
                outputs[key] = build_tensor_info(outputs[key])
            elif isinstance(outputs[key], dict):
                outputs[key] = {k: build_tensor_info(v) for k, v in outputs[key].items()}
            else:
                raise ValueError("Unexpected outputs type, only accept tf.Tensor or a dict as string to tf.Tensor map.")

        # detection only
        detection_only_inputs_tensor_info = {
            'inputs': inputs["detection_image_tensor"]
        }
        detection_only_outputs_tensor_info = {
            "boxes": outputs["detection_only_boxes"],
            "scores": outputs["detection_only_scores"],
            "classes": outputs["detection_only_classes"]
        }
        detection_only_signature = (
            tf.saved_model.signature_def_utils.build_signature_def(
                inputs=detection_only_inputs_tensor_info,
                outputs=detection_only_outputs_tensor_info,
                method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))
        # classify only
        classify_only_inputs_tensor_info = {
            'inputs': inputs["classify_image_tensor"]
        }
        classify_only_outputs_tensor_info = {
            "scores": outputs["classify_scores"],
            # "classes": outputs["classify_classes"]
        }
        classify_only_signature = (
            tf.saved_model.signature_def_utils.build_signature_def(
                inputs=classify_only_inputs_tensor_info,
                outputs=classify_only_outputs_tensor_info,
                method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))
        # merged detection classify
        detection_classify_inputs_tensor_info = {
            'inputs': inputs["detection_image_tensor"]
        }
        detection_classify_outputs_tensor_info = {
            "boxes": outputs["detection_only_boxes"],
            "scores": outputs["detection_classify_scores"],
            # "classes": outputs["detection_classify_classes"]
        }
        detection_classify_signature = (
            tf.saved_model.signature_def_utils.build_signature_def(
                inputs=detection_classify_inputs_tensor_info,
                outputs=detection_classify_outputs_tensor_info,
                method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))
        # class2index_map
        fake_inputs_tensor_info = {
            'inputs': inputs["fake_input"]
        }
        detection_class2index_map_signature = (
            tf.saved_model.signature_def_utils.build_signature_def(
                inputs=fake_inputs_tensor_info,
                outputs=outputs["detection_class_index_map"],
                method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))
        classify_class2index_map_signature = (
            tf.saved_model.signature_def_utils.build_signature_def(
                inputs=fake_inputs_tensor_info,
                outputs=outputs["classify_class_index_map"],
                method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))
        detection_classify_class2index_map_signature = (
            tf.saved_model.signature_def_utils.build_signature_def(
                inputs=fake_inputs_tensor_info,
                outputs=outputs["classify_class_index_map"],
                method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

        builder.add_meta_graph_and_variables(
            sess, [tf.saved_model.tag_constants.SERVING],
            signature_def_map={
                "detection_only": detection_only_signature,
                "classify_only": classify_only_signature,
                "detection_classify": detection_classify_signature,
                "detection_only_class2index_map": detection_class2index_map_signature,
                "classify_only_class2index_map": classify_class2index_map_signature,
                "detection_classify_class2index_map": detection_classify_class2index_map_signature
            },
        )
        builder.save()


def generator_random_label_json(file_path):
    random_dict={}
    for i in range(100):
        random_dict[str(i)]=i

    if not os.path.isfile(file_path):
        os.mknod(file_path)

    file_open = open(file_path, 'w')
    json.dump(random_dict, file_open)
    file_open.close()




def main(_):
    assert FLAG.detection_model_path != ""
    assert FLAG.classify_model_path != ""
    assert FLAG.export_dir != ""

    saved_model_path = os.path.join(FLAG.export_dir, str(FLAG.version))
    if FLAG.version == 0 and os.path.exists(saved_model_path):
        versions = [int(x) for x in os.listdir(FLAG.export_dir) if os.path.isdir(os.path.join(FLAG.export_dir, x))
                    and re.match("^\d+$", x)]
        FLAG.version = 1 + max(versions)
        saved_model_path = os.path.join(FLAG.export_dir, str(FLAG.version))
        print("Auto choose saved model version: %s" % FLAG.version)

    if os.path.exists(saved_model_path):
        raise Exception("The saved model version(%s) is already exists, please choose a bigger one." % FLAG.version)

    if os.path.isdir(FLAG.detection_model_path):
        FLAG.detection_model_path = os.path.join(FLAG.detection_model_path, "frozen_inference_graph.pb")
    if os.path.isdir(FLAG.classify_model_path):
        FLAG.classify_model_path = os.path.join(FLAG.classify_model_path, "frozen_inference_graph.pb")

    if FLAG.detection_label_index_map_path == "":
        FLAG.detection_label_index_map_path = \
            os.path.join(os.path.dirname(FLAG.detection_model_path), "label_index.map")
        generator_random_label_json(FLAG.detection_label_index_map_path)
    if FLAG.classify_label_index_map_path == "":
        FLAG.classify_label_index_map_path = \
            os.path.join(os.path.dirname(FLAG.classify_model_path), "label_index.map")
        generator_random_label_json(FLAG.classify_label_index_map_path)
    with open(FLAG.detection_label_index_map_path) as f:
        detection_label_index_map = json.load(f)
    with open(FLAG.classify_label_index_map_path) as f:
        classify_label_index_map = json.load(f)

    graph, inputs, outputs = build_merged_detection_classification(FLAG.detection_model_path,
                                                                   FLAG.classify_model_path,
                                                                   detection_label_index_map,
                                                                   classify_label_index_map)

    serialized_graph = graph.as_graph_def().SerializeToString()
    save_detection_classify_model(graph, inputs, outputs, saved_model_path)
    with tf.gfile.GFile(os.path.join(saved_model_path, "frozen_inference_graph.pb"), "wb") as f:
        f.write(serialized_graph)
    with open(os.path.join(saved_model_path, "label_index.map"), "w") as f:
        json.dump(classify_label_index_map, f, indent=4)
    print("Exported model to: %s" % saved_model_path)


if __name__ == '__main__':
    tf.app.run()
