
# activating the seatbelt detection conda env 
import subprocess
subprocess.call('conda activate seatbelt', shell=True)


import streamlit as st
import time
import cv2
import numpy as np
import tempfile
import os 



seatbelt_file_path ="C:/Users/shank/Desktop/aiml/ai_dashboard/seatbelt/YOLOR_PersonSeatbelt/"

current_dir = os.path.dirname(os.path.abspath(seatbelt_file_path))

# navigate to the parent directory
base_dir = os.path.dirname(current_dir)

# add the parent directory to the system path
import sys
sys.path.append(base_dir)

st.set_page_config(page_title="Seat Belt Detection", page_icon="🚗")

## Assign file paths here
weights_file_path = seatbelt_file_path + "weights/yolor_p6.pt"

@st.cache()
def load_model():
    return cv2.dnn.readNet(weights_file_path, seatbelt_file_path + 'yolov3.cfg')

sys.path.append('C:/Users/shank/Desktop/aiml/ai_dashboard/seatbelt/YOLOR_PersonSeatbelt/models')
from models.models import *
from utils.torch_utils import select_device, load_classifier, time_synchronized

def main():
    t0 = time.time()

    def load_classes(path):
        # Loads *.names file at 'path'
        with open(path, 'r') as f:
            names = f.read().split('\n')
        return list(filter(None, names))  # filter removes empty strings (such as last line)

    def draw_boxes(image_path, data):
        image = cv2.imread(image_path)
        output = image.copy()
        W = image.shape[1] 
        H = image.shape[0]
        data['xmin_norm'] = data['xc_norm']-data['w_norm']/2
        data['xmax_norm'] = data['xc_norm']+data['w_norm']/2
        data['ymin_norm'] = data['yc_norm'] - data['h_norm']/2
        data['ymax_norm'] = data['yc_norm'] + data['h_norm']/2
        data['xmin'] = data['xmin_norm']*W
        data['xmax'] = data['xmax_norm']*W
        data['ymin'] = data['ymin_norm']*H
        data['ymax'] = data['ymax_norm']*H
        for i in range(len(data)):
            startX = int(data.iloc[i]['xmin'])
            startY = int(data.iloc[i]['ymin'])
            endX   = int(data.iloc[i]['xmax'])
            endY   = int(data.iloc[i]['ymax'])
            text   = data.iloc[i]['Class']
            #text   = names2[int(data.iloc[i]['Class'])] 
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(output, (startX, startY), (endX, endY),(0, 0, 255), 2)
            cv2.putText(output, text, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return output


    images_path        = 'inference/images/'
    iou_bboxes_thres = 0
    tl = 3               # line thickness
    tf = max(tl - 1, 1)  # font thickness

    conf_thres        = 0.25
    iou_thres         = 0.2
    enable_GPU        = True
    view_img          = False
    imgsz             = 640

    # Initialize GPU
    if enable_GPU:
        device = select_device('0')
    else:
        device = select_device('cpu')
    half = device.type != 'cpu'  # half precision only supported on CUDA

    weights1          = 'yolor_p6.pt'
    cfg1              = 'cfg/yolor_p6.cfg'
    names1            = load_classes('data/coco.names')
    assigned_class_id1 = []
    # Load model1
    model1 = Darknet(cfg1, imgsz).cuda() #if you want cuda remove the comment
    model1.load_state_dict(torch.load(weights1, map_location=device)['model'])
    #mode1l = attempt_load(weights1, map_location=device)  # load FP32 model
    #imgsz = check_img_size(imgsz, s=model1.stride.max())  # check img_size
    model1.to(device).eval()
    if half:
        model1.half()  # to FP16


    weights2          = 'PersonSeatbelt/PersonSeatbelt_256_best_overall.pt'
    cfg2              = 'PersonSeatbelt/PersonSeatbelt_yolor_p6.cfg'
    names2            = load_classes('PersonSeatbelt/PersonSeatbelt_names.names')
    assigned_class_id2 = [1]
    
    # Load model2
    model2 = Darknet(cfg2, imgsz).cuda() #if you want cuda remove the comment
    model2.load_state_dict(torch.load(weights2, map_location=device)['model'])
    #mode12 = attempt_load(weights2, map_location=device)  # load FP32 model
    #imgsz = check_img_size(imgsz, s=model2.stride.max())  # check img_size
    model2.to(device).eval()
    if half:
        model2.half()  # to FP16


    t1 = time.time()
    PreviousTime = time.time()
    #print(t1-t0)
    ind =1
    for vid_name in os.listdir(images_path):
        df = pd.DataFrame(columns =["Class", "xc_norm", "yc_norm", "w_norm", "h_norm"])
        vid_name = images_path+vid_name
        txt_path           = 'inference/output/' + vid_name.split('/')[-1].split('.')[0]+'.txt'
        txt_out_path       = 'inference/output_det/' + vid_name.split('/')[-1].split('.')[0]+'.txt'
        out_path           = 'inference/output/' + vid_name.split('/')[-1]
        out_path_det1      = 'inference/output_det/detection_1_'+vid_name.split('/')[-1]
        out_path_det2      = 'inference/output_det/detection_2_'+vid_name.split('/')[-1]
        out_path_det_final = 'inference/output_det/final_' +vid_name.split('/')[-1]

###########
        # DETECTION Phase 1
        load_yolor_and_process_frame(model1, imgsz, device, vid_name, conf_thres, iou_thres, enable_GPU, view_img, assigned_class_id1, weights1, cfg1, names1)
        t2 = time.time()
        print()
        #print(t2-t1)
        if os.path.isfile(txt_path):
            df1 = pd.read_csv(txt_path, sep=" ", header=None)
            df1 = df1[[0,1,2,3,4]]
            df1.columns = ["Class", "xc_norm", "yc_norm", "w_norm", "h_norm"]
            df1['Class']= df1['Class'].map(lambda x: names1[x])
            df = pd.concat([df,df1], ignore_index = True)
        #print(df)
        t3 = time.time()
        #print(t3-t2)
###########
        #DETECTION Phase 2
        load_yolor_and_process_frame(model2, imgsz, device, vid_name, conf_thres, iou_thres, enable_GPU, view_img, assigned_class_id2, weights2, cfg2, names2)
        t4 = time.time()
        #print(t4-t3) 
        if os.path.isfile(txt_path):
            df2 = pd.read_csv(txt_path, sep=" ", header=None)
            df2 = df2[[0,1,2,3,4]]
            df2.columns = ["Class", "xc_norm", "yc_norm", "w_norm", "h_norm"]
            df2['Class']= df2['Class'].map(lambda x: names2[x])
            df = pd.concat([df, df2], ignore_index = True)
    
        #print('df :')
        #print(df)
        np.savetxt(txt_out_path, df.to_numpy(), fmt=["%s", "%f", "%f", "%f", "%f"])

        img_final = draw_boxes(vid_name, df)
        #cv2.imshow('img_final', img_final)
        cv2.imwrite(out_path_det_final, img_final)
        t5= time.time()
        #print(t5-t4)
        #print("time/inference :",ind, " : ", t5-t0)
        CurrentTime = time.time()
        print('time/inference', ind, ":", CurrentTime-PreviousTime)
        PreviousTime = CurrentTime
        ind = ind+1
    while (True):
        # Displays the window infinitely
        key = cv2.waitKey(0)
        # Shuts down the display window and terminates
        # the Python process when a specific key is
        # pressed on the window.
        # 27 is the esc key
        # 113 is the letter 'q'
        if key == 27 or key == 113:
            break
    print("total time ", ind, time.time()-to)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        main()
    except SystemExit:
        pass