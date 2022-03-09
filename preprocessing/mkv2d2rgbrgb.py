from argparse import ArgumentParser
from pyk4a import ImageFormat
import cv2
from typing import Optional, Tuple
from pyk4a import PyK4APlayback
import numpy as np
import pyk4a
import os
import re

def convert_to_bgra_if_required(color_format: ImageFormat, color_image):
    if color_format == ImageFormat.COLOR_MJPG:
        color_image = cv2.imdecode(color_image, cv2.IMREAD_COLOR)
    elif color_format == ImageFormat.COLOR_NV12:
        color_image = cv2.cvtColor(color_image, cv2.COLOR_YUV2BGRA_NV12)
        elif color_format == ImageFormat.COLOR_YUY2:
        color_image = cv2.cvtColor(color_image, cv2.COLOR_YUV2BGRA_YUY2)
    return color_image


def colorize(
    image: np.ndarray,
    clipping_range: Tuple[Optional[int], Optional[int]] = (None, None),
    colormap: int = cv2.COLORMAP_HSV,
) -> np.ndarray:
    if clipping_range[0] or clipping_range[1]:
        img = image.clip(clipping_range[0], clipping_range[1])
    else:
        img = image.copy()
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    img = cv2.applyColorMap(img, colormap)
    return img

def info(playback: PyK4APlayback):
    print(f"Record length: {playback.length / 1000000: 0.2f} sec")


def play(playback: PyK4APlayback,patha,pathc):
     num=1
     path=patha
     while True:
        try:
            capture = playback.get_next_capture()
            if capture.color is not None:
                
                cv2.imwrite(pathc+'/'+str(num)+'.jpg',convert_to_bgra_if_required(playback.configuration["color_format"], capture.color))
                
                
            
                if capture.depth is not None:

                    if pyk4a.depth_image_to_color_camera(capture.depth, playback.calibration, playback.thread_safe) is not None:
                       
                        cv2.imwrite(path+'/'+str(num)+'.png',pyk4a.depth_image_to_color_camera(capture.depth, playback.calibration, playback.thread_safe))
                num=num+1
            
        except EOFError:
            break
     cv2.destroyAllWindows()


def main() -> None:
    parser = ArgumentParser(description="pyk4a player")
    parser.add_argument("--seek", type=float, help="Seek file to specified offset in seconds", default=0.0)
    
    path=os.getcwd()
    videolist=[]
    for count in os.listdir(path):
        try:
            a=int(count[-1])
            videolist.append(count)
        except:
           None
    
    def tryint(s):                    
        try:
            return int(s)
        except ValueError:
            return s

    def str2int(v_str):            
        return [tryint(sub_str) for sub_str in re.split('([0-9]+)', v_str)]
    videolist.sort(key=str2int)


    for eachvideo in videolist:
        file=os.listdir(os.path.join(path,eachvideo))

        for mkvv in file:
            
                
            if mkvv[-3:]=='mkv':
                args = parser.parse_args()
                filename: str = os.path.join(path,eachvideo,mkvv)
                offset: float = args.seek
            
                playback = PyK4APlayback(filename)
                playback.open()
               
                info(playback)
            
                if offset != 0.0:
                    playback.seek(int(offset * 1000000))
                if os.path.exists(os.path.join(path,eachvideo,'d2rgb'))==0:
                    os.mkdir(os.path.join(path,eachvideo,'d2rgb'))
                else:
                    break
                if os.path.exists(os.path.join(path,eachvideo,'color'))==0:
                    os.mkdir(os.path.join(path,eachvideo,'color'))
                play(playback,os.path.join(path,eachvideo,'d2rgb'),os.path.join(path,eachvideo,'color'))
            
                playback.close()


if __name__ == "__main__":
    main()
