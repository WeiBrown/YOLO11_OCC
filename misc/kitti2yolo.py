# modify_annotations_txt.py
def category_combine():
    import glob
    import string
    txt_list = glob.glob('datasets/kitti_mini/labels/*.txt') # 存储Labels文件夹所有txt文件路径
    def show_category(txt_list):
        category_list= []
        for item in txt_list:
            try:
                with open(item) as tdf:
                    for each_line in tdf:
                        labeldata = each_line.strip().split(' ') # 去掉前后多余的字符并把其分开
                        category_list.append(labeldata[0]) # 只要第一个字段，即类别
            except IOError as ioerr:
                print('File error:'+str(ioerr))
        print(set(category_list)) # 输出集合
    
    def merge(line):
        each_line=''
        for i in range(len(line)):
            if i!= (len(line)-1):
                each_line=each_line+line[i]+' '
            else:
                each_line=each_line+line[i] # 最后一条字段后面不加空格
        each_line=each_line+'\n'
        return (each_line)
    
    print('before modify categories are:\n')
    show_category(txt_list)
    
    for item in txt_list:
        new_txt=[]
        try:
            with open(item, 'r') as r_tdf:
                for each_line in r_tdf:
                    labeldata = each_line.strip().split(' ')
                    if labeldata[0] in ['Truck','Van','Tram']: # 合并汽车类
                        labeldata[0] = labeldata[0].replace(labeldata[0],'Car')
                    if labeldata[0] == 'Person_sitting': # 合并行人类
                        labeldata[0] = labeldata[0].replace(labeldata[0],'Pedestrian')
                    if labeldata[0] == 'DontCare': # 忽略Dontcare类
                        continue
                    if labeldata[0] == 'Misc': # 忽略Misc类
                        continue
                    new_txt.append(merge(labeldata)) # 重新写入新的txt文件
            with open(item,'w+') as w_tdf: # w+是打开原文件将内容删除，另写新内容进去
                for temp in new_txt:
                    w_tdf.write(temp)
        except IOError as ioerr:
            print('File error:'+str(ioerr))
    
    print('\nafter modify categories are:\n')
    show_category(txt_list)

def calibration_convert():
    import os
    import cv2
    CLASS_MAPPING = {
        "Car": 0,
        "Van": 1,
        "Truck": 2,
        "Pedestrian": 3,
        "Person_sitting": 4,
        "Cyclist": 5,
        "Tram": 6,
        "Misc": 7,
    }

    def convert_kitti_to_yolo(kitti_label_dir, yolo_label_dir, images_dir):
        if not os.path.exists(yolo_label_dir):
            os.makedirs(yolo_label_dir)
        
        for label_file in os.listdir(kitti_label_dir):
            with open(os.path.join(kitti_label_dir, label_file), "r") as f:
                lines = f.readlines()

            image_file = label_file.replace(".txt", ".png")
            img = cv2.imread(os.path.join(images_dir, image_file))
            if img is None:
                print(f"img {image_file} not exists, skip")
                continue
            img_height, img_width, _ = img.shape

            yolo_lines = []
            for line in lines:
                parts = line.strip().split()
                obj_type, bbox_left, bbox_top, bbox_right, bbox_bottom = (
                    parts[0],
                    float(parts[4]),
                    float(parts[5]),
                    float(parts[6]),
                    float(parts[7]),
                )

                if obj_type not in CLASS_MAPPING:
                    continue

                class_id = CLASS_MAPPING[obj_type]
                x_center = ((bbox_left + bbox_right) / 2) / img_width
                y_center = ((bbox_top + bbox_bottom) / 2) / img_height
                width = (bbox_right - bbox_left) / img_width
                height = (bbox_bottom - bbox_top) / img_height

                yolo_line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                yolo_lines.append(yolo_line)

            with open(os.path.join(yolo_label_dir, label_file), "w") as f:
                f.write("\n".join(yolo_lines))
            print(f"trans {label_file} to YOLO format")

    kitti_label_dir = "datasets/kitti_mini/kitti_labels"
    yolo_label_dir = "datasets/kitti_mini/labels"
    images_dir = "datasets/kitti_mini/images"

    convert_kitti_to_yolo(kitti_label_dir, yolo_label_dir, images_dir)


calibration_convert()