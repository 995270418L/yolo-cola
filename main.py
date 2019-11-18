# '''
#     web 可视化测试模型
# '''
# import argparse
# import json
# import os
# import sys
# from os.path import join as opj
#
# from flask import Flask, request, Response
# import numpy as np
# import datetime
#
# sys.path.insert(0, opj(os.path.dirname(os.path.realpath(__file__)), '../'))
#
# project_dir = os.path.dirname(__file__)
# data_dir = os.path.join(project_dir, "data")
# model_path = os.path.join(project_dir, "model")
# pb_path = os.path.join(project_dir, "pb")
#
# class WebApplication():
#
#     def __init__(self):
#         self.model = os.path.join(model_path, "frozen_inference_graph.pb")
#         self.pbtxt = os.path.join(pb_path, "object_detection.pbtxt")
#         self.ini()
#
#     def ini(self):
#         # detection_graph = tf.Graph()
#         # with detection_graph.as_default():
#         #     od_graph_def = tf.GraphDef()
#         #     with tf.gfile.GFile(self.model, 'rb') as fid:
#         #         serialized_graph = fid.read()
#         #         od_graph_def.ParseFromString(serialized_graph)
#         #         tf.import_graph_def(od_graph_def, name='')
#         # config = tf.ConfigProto()
#         # config.gpu_options.allow_growth = True
#         # self.sess = tf.Session(graph=detection_graph, config=config)
#         # self.sess.graph.as_default()
#         # file = os.path.join(project_dir, "test.jpg")
#         # with open(file, 'rb') as f:
#         #     npimg = np.fromstring(f.read(), np.uint8)
#         #     img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
#         #     inp = cv2.resize(img, (450, 450))
#         #     inp = inp[:, :, [2, 1, 0]]  # BGR2RGB
#         #     self.sess.run([self.sess.graph.get_tensor_by_name('num_detections:0'),
#         #                          self.sess.graph.get_tensor_by_name('detection_scores:0'),
#         #                          self.sess.graph.get_tensor_by_name('detection_boxes:0'),
#         #                          self.sess.graph.get_tensor_by_name('detection_classes:0')],
#         #                         feed_dict={'image_tensor:0': inp.reshape(1, inp.shape[0], inp.shape[1], 3)})
#         # print("##################### system ini over ###########################")
#
#     def __call__(self, *args, **kwargs):
#         return self.upload()
#
#     def allowed_file(self, filename):
#         return True
#
#     def upload(self):
#         if request.method == 'POST':
#             file = request.files['file']
#             if file and self.allowed_file(file.filename):
#                 npimg = np.fromstring(file.read(), np.uint8)
#                 data = self.deal_image(npimg)
#                 res = {}
#                 res['data'] = data
#                 res['code'] = 0
#                 res['message'] = 'ok'
#                 print("response: {}".format(res))
#                 return Response(json.dumps(res), mimetype="application/json")
#         else:
#             res = {
#                 'code': 4,
#                 'message': '请求错误'
#             }
#             return Response(json.dumps(res), mimetype="application/json")
#
#     def deal_image(self, imgArr):
#         img = cv2.imdecode(imgArr, cv2.IMREAD_COLOR)
#         rows = img.shape[0]
#         cols = img.shape[1]
#         inp = cv2.resize(img, (450, 450))
#         inp = inp[:, :, [2, 1, 0]]  # BGR2RGB
#         oldTime= datetime.datetime.now()
#         out = self.sess.run([self.sess.graph.get_tensor_by_name('num_detections:0'),
#                     self.sess.graph.get_tensor_by_name('detection_scores:0'),
#                     self.sess.graph.get_tensor_by_name('detection_boxes:0'),
#                     self.sess.graph.get_tensor_by_name('detection_classes:0')],
#                    feed_dict={'image_tensor:0': inp.reshape(1, inp.shape[0], inp.shape[1], 3)})
#         endTime= datetime.datetime.now()
#         print("train an image cost time: {}".format(round((endTime - oldTime).microseconds / 1e3)))
#         num_detection = int(out[0][0])
#         data = []
#         if len(out) > 0:
#             for i in range(num_detection):
#                 output = {}
#                 output['cat'] = int(out[3][0][i])
#                 output['score'] = float(out[1][0][i])
#                 if round(output['score']) < 0.1:
#                     continue
#                 bbox = [float(v) for v in out[2][0][i]]
#                 x = int(bbox[1] * cols)
#                 y = int(bbox[0] * rows)
#                 right = int(bbox[3] * cols)
#                 bottom = int(bbox[2] * rows)
#                 output['box'] = [x, y, right, bottom] # 左上角， 右下角
#                 print("bbox :{}".format(output['box']))
#                 data.append(output)
#         return data
#
# class Ping(object):
#
#     def __init__(self):
#         pass
#
#     def __call__(self):
#         return "Hello, cola"
#
# class FlaskWrap(object):
#     def __init__(self, name):
#         self.app = Flask(name)
#         self.application = WebApplication()
#         self.ping = Ping()
#
#     def run(self):
#         self.app.run(host='0.0.0.0')
#
#     def add_endpoint(self, endpoint=None, endpoint_name=None):
#         self.app.add_url_rule(endpoint, endpoint_name, self.application, methods=['GET', 'POST'])
#
#     def add_ping(self, endpoint=None, endpoint_name=None):
#         self.app.add_url_rule(endpoint, endpoint_name, self.ping, methods=['GET', 'POST'])
#
# def main(argv=None):  # pylint: disable=unused-argument
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--image_folder", type=str, default="data/samples", help="path to dataset")
#     parser.add_argument("--model_def", type=str, default="config/yolov3-cola.cfg", help="path to model definition file")
#     parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="path to weights file")
#     parser.add_argument("--class_path", type=str, default="data/cola/cola.names", help="path to class label file")
#     parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
#     # parser.add_argument("--conf_thres", type=float, default=0.1, help="object confidence threshold")
#     parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
#     # parser.add_argument("--nms_thres", type=float, default=0.1, help="iou thresshold for non-maximum suppression")
#     parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
#     parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
#     parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
#     parser.add_argument("--checkpoint_model", type=str, help="path to checkpoint model")
#     opt = parser.parse_args()
#     print(opt)
#     app = FlaskWrap(__name__)
#     app.add_endpoint(endpoint="/upload", endpoint_name="ad")
#     app.add_ping(endpoint="/ping", endpoint_name="ping")
#     app.run()
#
# if __name__ == '__main__':
#     main()