import numpy as np
import cv2 as cv
import os
from PIL import Image


yolo_dir = '/home/hessesummer/github/NTS-Net-my/yolov3'  # YOLO文件路径
CONFIDENCE = 0.5  # 过滤弱检测的最小概率
THRESHOLD = 0.1  # 非最大值抑制阈值
weightsPath = os.path.join(yolo_dir, 'yolov3.weights')
configPath = os.path.join(yolo_dir, 'yolov3.cfg')
labelsPath = os.path.join(yolo_dir, 'coco.names')
default_imgPath = os.path.join(yolo_dir, 'test.jpg')

# 得到labels列表
with open(labelsPath, 'rt') as f:
    labels = f.read().rstrip('\n').split('\n')


def getbbox(pilimg):
    """
    :param img: RGB格式的img
    :return: img(BGR格式), boxes, confidences, classIDs, idxs
    """
    # 加载网络、配置权重
    net = cv.dnn.readNetFromDarknet(configPath, weightsPath)
    print("[INFO] loading YOLO from disk...")

    # 加载图片、转为blob格式、送入网络输入层
    img = cv.cvtColor(np.array(pilimg), cv.COLOR_RGB2BGR)
    if img is None:
        raise Exception('图片路径错误，读入失败')
    blobImg = cv.dnn.blobFromImage(img, 1.0/255.0, (416, 416), None, True, False)
    net.setInput(blobImg)

    # 获取网络输出层信息（所有输出层的名字），设定并前向传播
    outInfo = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(outInfo)  # 得到各个输出层的各个框框，是二维结构

    (H, W) = img.shape[:2]  # 拿到图片宽高，下面用
    # 过滤outs[center_x, center_y, width, height, objectness, N-class score data]，过滤后的结果放入：
    boxes = []
    confidences = []
    classIDs = []
    # # 1）过滤掉置信度低的框框
    for out in layerOutputs:  # 各个输出层
        for detection in out:  # 各个框框
            # 拿到置信度
            scores = detection[5:]  # 各个类别的置信度
            classID = np.argmax(scores)  # 最高置信度的id即为分类id
            confidence = scores[classID]  # 拿到置信度

            # 根据置信度筛查。并且加上自定义筛查1：只选择鸟框框
            if labels[classID] == 'bird' and confidence > CONFIDENCE:   # 只选择鸟框框
                box = detection[0:4] * np.array([W, H, W, H])  # 将边界框放回图片尺寸
                # 扩大边界框，可能会超图像边界
                box[2] = box[2] * 1.5
                box[3] = box[3] * 1.5
                # 获得x1、x2
                (centerX, centerY, width, height) = box.astype("int")
                x1 = max(0, int(centerX - (width / 2)))
                y1 = max(0, int(centerY - (height / 2)))
                # 获得x2、y2
                x2 = min(W, x1 + int(width))
                y2 = min(H, y1 + int(height))
                boxes.append([x1, y1, x2, y2])

                confidences.append(float(confidence))
                classIDs.append(classID)

    # # 2）应用非最大值抑制(non-maxima suppression，nms)进一步筛掉
    idxs = cv.dnn.NMSBoxes(boxes, confidences, CONFIDENCE, THRESHOLD)  # boxes中，保留的box的索引index存入idxs
    idxs = np.array(idxs).flatten()

    # # 3）自定义筛查2：只选择置信度最大的那一个框框
    if len(idxs) > 0 and len(confidences) > 0:  # 如果有识别出目标
        boxes = np.array(boxes)[idxs]
        confidences = np.array(confidences)[idxs]
        classIDs = np.array(classIDs)[idxs]
        top_idx = np.argmax(confidences)
        box = boxes[top_idx]
        confidence = confidences[top_idx]
        classID = classIDs[top_idx]

    else:
        box = np.array([])
        confidence = None
        classID = None
        print('没有识别出目标')

    # print(box) 是一个Ndarray，其中有四个元素x1, y1, x2, y2
    return img, box, confidence, classID


def showRes(img, box, confidence, classID):

    # 应用检测结果
    if len(box) > 0:
        np.random.seed(42)
        COLORS = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")  # 框框显示颜色，每一类有不同的颜色，每种颜色都是由RGB三个值组成的，所以size为
        x1, y1, x2, y2 = box
        color = [int(c) for c in COLORS[classID]]
        cv.rectangle(img, (x1, y1), (x2, y2), color, 2)  # 线条粗细为2px
        text = "{}: {:.4f}".format(labels[classID], confidence)
        cv.putText(img, text, (x1, y1-5), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)  # 字体风格、字体大小、粗细
    cv.imshow('image', img)
    cv.waitKey(0)

if __name__ == '__main__':
    imgPath = os.path.join(yolo_dir,
                           'testfail.jpg')
    img = Image.open(imgPath).convert('RGB')
    img, box, confidence, classID = getbbox(img)
    showRes(img, box, confidence, classID)
