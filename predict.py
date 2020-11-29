{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [
    {
     "ename": "ImportError",
     "evalue": "attempted relative import with no known parent package",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mImportError\u001b[0m                               Traceback (most recent call last)",
      "\u001b[1;32m<ipython-input-25-953aa3dd6da9>\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[0;32m      2\u001b[0m \u001b[1;32mimport\u001b[0m \u001b[0mtime\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      3\u001b[0m \u001b[1;32mimport\u001b[0m \u001b[0mnumpy\u001b[0m \u001b[1;32mas\u001b[0m \u001b[0mnp\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m----> 4\u001b[1;33m \u001b[1;32mfrom\u001b[0m \u001b[1;33m.\u001b[0m\u001b[0myolo\u001b[0m \u001b[1;32mimport\u001b[0m \u001b[0mYOLO\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m      5\u001b[0m \u001b[1;32mfrom\u001b[0m \u001b[0mPIL\u001b[0m \u001b[1;32mimport\u001b[0m \u001b[0mImage\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;31mImportError\u001b[0m: attempted relative import with no known parent package"
     ]
    }
   ],
   "source": [
    "import cv2\n",
    "import time\n",
    "import numpy as np\n",
    "from yolo import YOLO\n",
    "from PIL import Image"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "#-------------------------------------#\n",
    "#       对单张图片进行预测\n",
    "#-------------------------------------#\n",
    "def detect_image(image_path):\n",
    "    print('Start detect!')\n",
    "    yolo = YOLO()\n",
    "    try:\n",
    "        image = Image.open(image_path)\n",
    "    except:\n",
    "        print('Open Error! Try again!')\n",
    "        pass\n",
    "    else:\n",
    "        r_image = yolo.detect_image(image)\n",
    "        r_image.save(image_path.split('.')[0] + '_result.png')\n",
    "    print('Finish detect!')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#-------------------------------------#\n",
    "#       对一段视频进行检测\n",
    "#-------------------------------------#\n",
    "def detect_video():\n",
    "    print('Start detect!')\n",
    "    yolo = YOLO()\n",
    "    capture = cv2.VideoCapture(0)\n",
    "    writer = None\n",
    "    fps = 0.0\n",
    "    while True:\n",
    "        t1 = time.time()\n",
    "        # 读取某一帧\n",
    "        grabbed, frame = capture.read()\n",
    "        if not grabbed:\n",
    "            break\n",
    "        # opencv读取的是BGR，格式转变，BGRtoRGB\n",
    "        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)\n",
    "        # 转变成Image\n",
    "        frame = Image.fromarray(np.uint8(frame))\n",
    "        # 进行检测\n",
    "        frame = np.array(yolo.detect_image(frame))\n",
    "        # RGBtoBGR满足opencv显示格式\n",
    "        frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)\n",
    "\n",
    "        fps  = (fps + (1. / (time.time() - t1))) / 2\n",
    "        print(\"FPS: %.2f\" % (fps))\n",
    "        frame = cv2.putText(frame, \"FPS: %.2f\" % (fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)\n",
    "        \n",
    "        if writer is None:\n",
    "            fourcc = cv2.VideoWriter_fourcc(*'MP4V')\n",
    "            writer = cv2.VideoWriter(video_path.split('.')[0] + '_result.mp4', fourcc, 30, (frame.shape[1], frame.shape[0]), True)\n",
    "        writer.write(frame)\n",
    "    writer.release()\n",
    "    capture.release()\n",
    "    print('Finish detect!')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Start detect!\n",
      "Loading pretrained weights.\n",
      "Finish loading!\n",
      "model_data/yolov4_maskdetect_weights1.pth model, anchors, and classes loaded.\n",
      "(5, 4)\n",
      "b'mask: 0.95'\n",
      "b'mask: 0.84'\n",
      "b'mask: 0.88'\n",
      "b'mask: 0.80'\n",
      "b'nomask: 0.96'\n",
      "Finish detect!\n"
     ]
    }
   ],
   "source": [
    "detect_image('testImage.png')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Start detect!\n"
     ]
    },
    {
     "ename": "NameError",
     "evalue": "name 'YOLO' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "\u001b[1;32m<ipython-input-13-a775d0dd8928>\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[1;32m----> 1\u001b[1;33m \u001b[0mdetect_video\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m",
      "\u001b[1;32m<ipython-input-5-eaebfa9bd071>\u001b[0m in \u001b[0;36mdetect_video\u001b[1;34m()\u001b[0m\n\u001b[0;32m      4\u001b[0m \u001b[1;32mdef\u001b[0m \u001b[0mdetect_video\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      5\u001b[0m     \u001b[0mprint\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;34m'Start detect!'\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m----> 6\u001b[1;33m     \u001b[0myolo\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mYOLO\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m      7\u001b[0m     \u001b[0mcapture\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mcv2\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mVideoCapture\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;36m0\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      8\u001b[0m     \u001b[0mwriter\u001b[0m \u001b[1;33m=\u001b[0m \u001b[1;32mNone\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;31mNameError\u001b[0m: name 'YOLO' is not defined"
     ]
    }
   ],
   "source": [
    "detect_video()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.8"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
