import argparse
import time
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
from scipy.spatial import distance as dist
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, non_max_suppression, apply_classifier, scale_coords, \
    xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

from torchvision import transforms


class V5:
    def __init__(self , weights):
        # Initialize
        set_logging()
        device = select_device('')
        half = device.type != 'cpu'  # half precision only supported on CUDA
        
        # Load model
        self.model = attempt_load(weights, map_location=device)  # load FP32 model

    def detect(self , img1, area_thres :int =10000, height_thres: int =1000, width_thres: int=700):
        
        # Get names and colors
        names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

        trans = transforms.ToTensor()   
        # while True:
        # Run inference
        t0 = time.time()

        # img1 = cv2.imread('data/images/1.jpg')
        img1 = cv2.cvtColor(img1 , cv2.COLOR_BGR2RGB)
        img  = trans(img1)
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = self.model(img, augment=False)[0]
        # Apply NMS
        pred = non_max_suppression(pred, 0.3, 0.45, classes=None, agnostic=False)
        t2 = time_synchronized()

        # Process detections
        for i, det in enumerate(pred):  
            print(det)
            gn = torch.tensor(img1.shape)[[1, 0, 1, 0]]  # norma
            t=0
            Nc=0
            display_str_list = []
            display_str_dict={}
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img1.shape).round()
            

                for *xyxy, conf, cls in reversed(det):
                    # print('xyxy = ',xyxy , 'conf = ',conf , 'cls = ',cls)
                    label = f'{names[int(cls)]} {conf:.2f}'
                    x1=int(xyxy[0].item())
                    y1=int(xyxy[1].item())
                    x2=int(xyxy[2].item())
                    y2=int(xyxy[3].item())


                    
                    display_str_dict = {'name':names[int(cls)] , 'score':f'{conf:.2f}'}
                    display_str_dict['ymin'] = y1
                    display_str_dict['xmin'] = x1
                    display_str_dict['ymax'] = y2
                    display_str_dict['xmax'] = x2
                    display_str_dict['area'] = (x2 - x1) * (y2 - y1)

                    if display_str_dict['name'] == "Torn":
                        t = t +1
                    elif display_str_dict['name'] == "notClean" : 
                        Nc = Nc + 1
                    
                    display_str_list.append(display_str_dict)
                    # cv2.rectangle(img1 , (x1,y1) ,(x2,y2) , (255,0,0),3  )
                    # cv2.putText(img1, f'{label} {conf:.2f}', (x1 , y1-10), cv2.FONT_HERSHEY_SIMPLEX , 1, (255,0,0), 3, cv2.LINE_AA)       
                    plot_one_box(xyxy, img1, label=label, color=colors[int(cls)], line_thickness=3)
            
            # Rejected = False ; Rejected_height = False
            # for i in range(len(display_str_list)):
            #     if display_str_list[i]['name']=='Top':
            #         height = display_str_list[i]['ymax'] - display_str_list[i]['ymin']
            #         width = display_str_list[i]['xmax'] - display_str_list[i]['xmin']
            #         width = abs(width)
            #         if (height < 1000) or (width < 700):
            #             Rejected_height = True
            #         continue
            #     else:
            #         area_detected = display_str_list[i]['area']
            #         print(area_detected)
            #         if area_detected > 2000:
            #             Rejected = True
            #             if Rejected:
            #                 break
            self.display_str_list = display_str_list
            Rejected , Rejected_height = self.height_area_Rejected()
            if (Rejected or Rejected_height):
                rejected_final = True
            print(Rejected , Rejected_height , rejected_final)
            print(f'Done. ({t2 - t1:.3f}s)')
            print(display_str_list)
            # cv2.imshow(str('a'), img1)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            self.height_area_Rejected()
            return img1, display_str_list ,t, Nc , rejected_final
    
    def midpoint(self, ptA, ptB):
        return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

    def height_area_Rejected(self, area_thres :int =10000, height_thres: int =1000, width_thres: int=700):
            Rejected = False ; Rejected_height = False
            display_str_list = self.display_str_list
            dimA , dimB = None , None
            for i in range(len(display_str_list)):
                if display_str_list[i]['name']=='Top':
                    tl = np.array([display_str_list[i]['xmin'] , display_str_list[i]['ymax']], dtype='i')
                    tr = np.array([display_str_list[i]['xmax'] , display_str_list[i]['ymax']], dtype='i')
                    bl = np.array([display_str_list[i]['xmin'] , display_str_list[i]['ymin']], dtype='i')
                    br = np.array([display_str_list[i]['xmax'] , display_str_list[i]['ymin']], dtype='i')
                    print(f'im printing {tl}, {tr}, {bl}, {br}')
                    (tltrX, tltrY) = self.midpoint(tl, tr)
                    (blbrX, blbrY) = self.midpoint(bl, br)

                    (tlblX, tlblY) = self.midpoint(tl, bl)
                    (trbrX, trbrY) = self.midpoint(tr, br)

                    # compute the Euclidean distance between the midpoints
                    dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
                    dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

                    # USED FOR CALLIBERATION
                    # if pixelsPerMetric is None:
                    #pixelsPerMetric = dB / width

                    pixelsPerMetric = 90.33
                    dimA = dA / pixelsPerMetric
                    dimB = dB / pixelsPerMetric

                    dimA = dA / pixelsPerMetric
                    dimB = dB / pixelsPerMetric
                    dimA = dimA*(25.4)
                    dimB = dimB*(25.4)
                    if (dimA < 20) or (dimB < 20):
                        Rejected_height = True
                        continue
                else:
                    area_detected = display_str_list[i]['area']
                    print(area_detected)
                    if area_detected > 2000:
                        Rejected = True
                        if Rejected:
                            break

            if dimA is not None:  
                print(f'Dimensions are {dimA}, {dimB}')
            return Rejected , Rejected_height

                    




if __name__ == '__main__':
    detect()
