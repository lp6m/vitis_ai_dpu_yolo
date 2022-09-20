#include <glog/logging.h>
#include <google/protobuf/text_format.h>

#include <cmath>
#include <iostream>
#include <numeric>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vitis/ai/dpu_task.hpp>
#include <vitis/ai/nnpp/yolov3.hpp>
#include <fstream>
#include <map>
#include <vector>
#include <math.h>

using namespace std;
using namespace cv;

// The parameters of yolov3_voc, each value could be set as actual needs.
// Such format could be refer to the prototxts in /etc/dpu_model_param.d.conf/.

const string readFile(const char *filename){
  ifstream ifs(filename);
  return string(istreambuf_iterator<char>(ifs),
                istreambuf_iterator<char>());
}

class YoloRunner{
  public:
    unique_ptr<vitis::ai::DpuTask> task;
    vitis::ai::proto::DpuModelParam modelconfig;
    cv::Size model_input_size;
    vector<vitis::ai::library::InputTensor> input_tensor;
    struct bbox{
      int label;
      float xmin;
      float ymin;
      float width;
      float height;
      float score;
      bbox(vitis::ai::YOLOv3Result::BoundingBox yolobbox, float img_width, float img_height){
        this->label = yolobbox.label;
        this->score = yolobbox.score;
        // does not clamp here
        this->xmin = yolobbox.x * img_width;
        this->ymin = yolobbox.y * img_height;
        this->width = yolobbox.width * img_width;
        this->height = yolobbox.height * img_height;
      }
    };

  public: YoloRunner(const char* modelconfig_path, const char* modelfile_path){
    const string config_str = readFile(modelconfig_path);
    auto ok = google::protobuf::TextFormat::ParseFromString(config_str, &(this->modelconfig));
    if (!ok) {
      cerr << "Set parameters failed!" << endl;
      abort();
    }
    this->task = vitis::ai::DpuTask::create(modelfile_path);
    this->input_tensor = task->getInputTensor(0u);
    int width = this->input_tensor[0].width;
    int height = this->input_tensor[0].height;
    this->model_input_size = cv::Size(width, height);
    this->task->setMeanScaleBGR({0.0f, 0.0f, 0.0f},
                        {0.00390625f, 0.00390625f, 0.00390625f});
  }
  private: cv::Mat Preprocess(cv::Mat img){
    cv::Mat resized_img;
    cv::resize(img, resized_img, this->model_input_size);
    return resized_img;
  }
  public: vector<bbox> Run(cv::Mat img){
    cv::Mat resized_img = this->Preprocess(img);
    vector<int> input_cols = {img.cols};
    vector<int> input_rows = {img.rows};
    vector<cv::Mat> inputs = {resized_img};
    task->setImageRGB(inputs);
    task->run(0);

    auto output_tensor = task->getOutputTensor(0u);
    auto results = vitis::ai::yolov3_post_process(
        input_tensor, output_tensor, this->modelconfig, input_cols, input_rows);
    auto result = results[0]; //batch_size is 1
    vector<bbox> bboxes;
    for(auto& yolobbox: result.bboxes){
      bboxes.push_back(bbox(yolobbox, img.cols, img.rows));
    }
    return bboxes;
  }

};

std::string get_basename(std::string& path) {
  int l = path.find_last_of('/')+1;
  int r = path.find_last_of('.');
    return path.substr(l, r-l);
}

map<string, string> bbox_to_map(YoloRunner::bbox bbox, int frame_id){
  map<string, string> res;
  res["frame_id"] = to_string(frame_id);
  res["prob"] = to_string(bbox.score);
  res["x"] = to_string(bbox.xmin);
  res["y"] = to_string(bbox.ymin);
  res["width"] = to_string(bbox.width);
  res["height"] = to_string(bbox.height);
  return res;
}

int main(int argc, char* argv[]) {
  if (argc != 5) {
    cerr << "usage ./a.out config(.prototxt) modelfile(.xmodel) image(.jpg) image" << endl;
  }
  char* configfile  = argv[1];
  char* modelfile = argv[2];
  string img_or_video_file = string(argv[3]);

  cout << configfile << " " << modelfile << " " << img_or_video_file;
  auto runner = YoloRunner(configfile, modelfile);
  cout << "Model Initialize Done" << endl;
  std::string img_or_video_mode = std::string(argv[4]);
  if (img_or_video_mode == "image") {
    cv::Mat img = cv::imread(img_or_video_file);
    vector<YoloRunner::bbox> bboxes = runner.Run(img);
    string label_names[] = {"person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"};
    for (auto& box : bboxes) {
      int label = box.label;
      float confidence = box.score;
      float xmin = max(0.0f, box.xmin);
      float ymin = max(0.0f, box.ymin);
      float xmax = min(box.xmin + box.width, (float)img.cols-1.0f);
      float ymax = min(box.ymin + box.height, (float)img.rows-1.0f);
      cout << label_names[box.label] << " " << box.score << " " << xmin << " " << xmax << " " << ymin << " " << ymax << endl;
      rectangle(img, Point(xmin, ymin), Point(xmax, ymax),
                Scalar(0, 255, 0), 3, 1, 0);
    }
    imwrite("result.jpg", img);
  } else {
    cerr << "unknown mode :" << img_or_video_mode << endl;
  }


  return 0;
}
