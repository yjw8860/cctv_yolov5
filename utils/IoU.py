import glob
import json
import os
import shutil
import operator
import sys
import math
import argparse

import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt

MINOVERLAP = 0.5 # default value (defined in the PASCAL VOC2012 challenge)

def log_average_miss_rate(prec, rec, num_images):
    """
        log-average miss rate:
            Calculated by averaging miss rates at 9 evenly spaced FPPI points
            between 10e-2 and 10e0, in log-space.

        output:
                lamr | log-average miss rate
                mr | miss rate
                fppi | false positives per image

        references:
            [1] Dollar, Piotr, et al. "Pedestrian Detection: An Evaluation of the
               State of the Art." Pattern Analysis and Machine Intelligence, IEEE
               Transactions on 34.4 (2012): 743 - 761.
    """

    # if there were no detections of that class
    if prec.size == 0:
        lamr = 0
        mr = 1
        fppi = 0
        return lamr, mr, fppi

    fppi = (1 - prec)
    mr = (1 - rec)

    fppi_tmp = np.insert(fppi, 0, -1.0)
    mr_tmp = np.insert(mr, 0, 1.0)

    # Use 9 evenly spaced reference points in log-space
    ref = np.logspace(-2.0, 0.0, num = 9)
    for i, ref_i in enumerate(ref):
        # np.where() will always find at least 1 index, since min(ref) = 0.01 and min(fppi_tmp) = -1.0
        j = np.where(fppi_tmp <= ref_i)[-1][-1]
        ref[i] = mr_tmp[j]

    # log(0) is undefined, so we use the np.maximum(1e-10, ref)
    lamr = math.exp(np.mean(np.log(np.maximum(1e-10, ref))))

    return lamr, mr, fppi

"""
 throw error and exit
"""
def error(msg):
    print(msg)
    sys.exit(0)

"""
 check if the number is a float between 0.0 and 1.0
"""
def is_float_between_0_and_1(value):
    try:
        val = float(value)
        if val > 0.0 and val < 1.0:
            return True
        else:
            return False
    except ValueError:
        return False

"""
 Calculate the AP given the recall and precision array
    1st) We compute a version of the measured precision/recall curve with
         precision monotonically decreasing
    2nd) We compute the AP as the area under this curve by numerical integration.
"""
def voc_ap(rec, prec):
    """
    --- Official matlab code VOC2012---
    mrec=[0 ; rec ; 1];
    mpre=[0 ; prec ; 0];
    for i=numel(mpre)-1:-1:1
            mpre(i)=max(mpre(i),mpre(i+1));
    end
    i=find(mrec(2:end)~=mrec(1:end-1))+1;
    ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    """
    rec.insert(0, 0.0) # insert 0.0 at begining of list
    rec.append(1.0) # insert 1.0 at end of list
    mrec = rec[:]
    prec.insert(0, 0.0) # insert 0.0 at begining of list
    prec.append(0.0) # insert 0.0 at end of list
    mpre = prec[:]
    """
     This part makes the precision monotonically decreasing
        (goes from the end to the beginning)
        matlab: for i=numel(mpre)-1:-1:1
                    mpre(i)=max(mpre(i),mpre(i+1));
    """
    # matlab indexes start in 1 but python in 0, so I have to do:
    #     range(start=(len(mpre) - 2), end=0, step=-1)
    # also the python function range excludes the end, resulting in:
    #     range(start=(len(mpre) - 2), end=-1, step=-1)
    for i in range(len(mpre)-2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i+1])
    """
     This part creates a list of indexes where the recall changes
        matlab: i=find(mrec(2:end)~=mrec(1:end-1))+1;
    """
    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i-1]:
            i_list.append(i) # if it was matlab would be i + 1
    """
     The Average Precision (AP) is the area under the curve
        (numerical integration)
        matlab: ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    """
    ap = 0.0
    for i in i_list:
        ap += ((mrec[i]-mrec[i-1])*mpre[i])
    return ap, mrec, mpre


"""
 Convert the lines of a file to a list
"""
def file_lines_to_list(path):
    # open txt file lines to a list
    with open(path) as f:
        content = f.readlines()
    # remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content]
    return content

"""
 Draws text in image
"""
def draw_text_in_image(img, text, pos, color, line_width):
    font = cv2.FONT_HERSHEY_PLAIN
    fontScale = 1
    lineType = 1
    bottomLeftCornerOfText = pos
    cv2.putText(img, text,
            bottomLeftCornerOfText,
            font,
            fontScale,
            color,
            lineType)
    text_width, _ = cv2.getTextSize(text, font, fontScale, lineType)[0]
    return img, (line_width + text_width)

"""
 Plot - adjust axes
"""
def adjust_axes(r, t, fig, axes):
    # get text width for re-scaling
    bb = t.get_window_extent(renderer=r)
    text_width_inches = bb.width / fig.dpi
    # get axis width in inches
    current_fig_width = fig.get_figwidth()
    new_fig_width = current_fig_width + text_width_inches
    propotion = new_fig_width / current_fig_width
    # get axis limit
    x_lim = axes.get_xlim()
    axes.set_xlim([x_lim[0], x_lim[1]*propotion])

"""
 Draw plot using Matplotlib
"""
def draw_plot_func(dictionary, n_classes, window_title, plot_title, x_label, output_path, to_show, plot_color, true_p_bar):
    # sort the dictionary by decreasing value, into a list of tuples
    sorted_dic_by_value = sorted(dictionary.items(), key=operator.itemgetter(1))
    # unpacking the list of tuples into two lists
    sorted_keys, sorted_values = zip(*sorted_dic_by_value)
    #
    if true_p_bar != "":
        """
         Special case to draw in:
            - green -> TP: True Positives (object detected and matches ground-truth)
            - red -> FP: False Positives (object detected but does not match ground-truth)
            - pink -> FN: False Negatives (object not detected but present in the ground-truth)
        """
        fp_sorted = []
        tp_sorted = []
        for key in sorted_keys:
            fp_sorted.append(dictionary[key] - true_p_bar[key])
            tp_sorted.append(true_p_bar[key])
        plt.barh(range(n_classes), fp_sorted, align='center', color='crimson', label='False Positive')
        plt.barh(range(n_classes), tp_sorted, align='center', color='forestgreen', label='True Positive', left=fp_sorted)
        # add legend
        plt.legend(loc='lower right')
        """
         Write number on side of bar
        """
        fig = plt.gcf() # gcf - get current figure
        axes = plt.gca()
        r = fig.canvas.get_renderer()
        for i, val in enumerate(sorted_values):
            fp_val = fp_sorted[i]
            tp_val = tp_sorted[i]
            fp_str_val = " " + str(fp_val)
            tp_str_val = fp_str_val + " " + str(tp_val)
            # trick to paint multicolor with offset:
            # first paint everything and then repaint the first number
            t = plt.text(val, i, tp_str_val, color='forestgreen', va='center', fontweight='bold')
            plt.text(val, i, fp_str_val, color='crimson', va='center', fontweight='bold')
            if i == (len(sorted_values)-1): # largest bar
                adjust_axes(r, t, fig, axes)
    else:
        plt.barh(range(n_classes), sorted_values, color=plot_color)
        """
         Write number on side of bar
        """
        fig = plt.gcf() # gcf - get current figure
        axes = plt.gca()
        r = fig.canvas.get_renderer()
        for i, val in enumerate(sorted_values):
            str_val = " " + str(val) # add a space before
            if val < 1.0:
                str_val = " {0:.2f}".format(val)
            t = plt.text(val, i, str_val, color=plot_color, va='center', fontweight='bold')
            # re-set axes to show number inside the figure
            if i == (len(sorted_values)-1): # largest bar
                adjust_axes(r, t, fig, axes)
    # set window title
    fig.canvas.set_window_title(window_title)
    # write classes in y axis
    tick_font_size = 12
    plt.yticks(range(n_classes), sorted_keys, fontsize=tick_font_size)
    """
     Re-scale height accordingly
    """
    init_height = fig.get_figheight()
    # comput the matrix height in points and inches
    dpi = fig.dpi
    height_pt = n_classes * (tick_font_size * 1.4) # 1.4 (some spacing)
    height_in = height_pt / dpi
    # compute the required figure height
    top_margin = 0.15 # in percentage of the figure height
    bottom_margin = 0.05 # in percentage of the figure height
    figure_height = height_in / (1 - top_margin - bottom_margin)
    # set new height
    if figure_height > init_height:
        fig.set_figheight(figure_height)

    # set plot title
    plt.title(plot_title, fontsize=14)
    # set axis titles
    # plt.xlabel('classes')
    plt.xlabel(x_label, fontsize='large')
    # adjust size of window
    fig.tight_layout()
    # save the plot
    fig.savefig(output_path)
    # show image
    if to_show:
        plt.show()
    # close the plot
    plt.close()


def get_IoU(DIR):
    mapparser = argparse.ArgumentParser()
    mapparser.add_argument('-na', '--no-animation', help="no animation is shown.", action="store_true")
    mapparser.add_argument('-np', '--no-plot', help="no plot is shown.", action="store_true")
    mapparser.add_argument('-q', '--quiet', help="minimalistic console output.", action="store_true")
    # argparse receiving list of classes to be ignored (e.g., python main.py --ignore person book)
    mapparser.add_argument('-i', '--ignore', nargs='+', type=str, help="ignore a list of classes.")
    # argparse receiving list of classes with specific IoU (e.g., python main.py --set-class-iou person 0.7)
    mapparser.add_argument('--set-class-iou', nargs='+', type=str, help="set IoU for a specific class.")
    args = mapparser.parse_args()
    # if there are no classes to ignore then replace None by empty list
    if args.ignore is None:
        args.ignore = []

    specific_iou_flagged = False
    if args.set_class_iou is not None:
        specific_iou_flagged = True

    GT_PATH = os.path.join(DIR['ROOT_DIR'], 'labels')
    DR_PATH = os.path.join(DIR['ROOT_DIR'], 'results')
    # if there are no images then no animation can be shown
    IMG_PATH = os.path.join(DIR['ROOT_DIR'], 'images')
    if os.path.exists(IMG_PATH):
        for dirpath, dirnames, files in os.walk(IMG_PATH):
            if not files:
                # no image files found
                args.no_animation = True
    else:
        args.no_animation = True

    # try to import OpenCV if the user didn't choose the option --no-animation
    show_animation = False
    if not args.no_animation:
        try:
            import cv2
            show_animation = True
        except ImportError:
            print("\"opencv-python\" not found, please install to visualize the results.")
            args.no_animation = True

    # try to import Matplotlib if the user didn't choose the option --no-plot
    draw_plot = False
    if not args.no_plot:
        try:
            import matplotlib.pyplot as plt
            draw_plot = True
        except ImportError:
            print("\"matplotlib\" not found, please install it to get the resulting plots.")
            args.no_plot = True
    """
     Create a ".temp_files/" and "output/" directory
    """
    TEMP_FILES_PATH = ".temp_files"
    if not os.path.exists(TEMP_FILES_PATH):  # if it doesn't exist already
        os.makedirs(TEMP_FILES_PATH)
    output_files_path = "output"
    if os.path.exists(output_files_path):  # if it exist already
        # reset the output directory
        shutil.rmtree(output_files_path)

    os.makedirs(output_files_path)
    if draw_plot:
        os.makedirs(os.path.join(output_files_path, "classes"))
    if show_animation:
        os.makedirs(os.path.join(output_files_path, "images", "detections_one_by_one"))

    """
     ground-truth
         Load each of the ground-truth files into a temporary ".json" file.
         Create a list of all the class names present in the ground-truth (gt_classes).
    """
    # get a list with the ground-truth files
    ground_truth_files_list = glob.glob(GT_PATH + '/*.txt')
    if len(ground_truth_files_list) == 0:
        error("Error: No ground-truth files found!")
    ground_truth_files_list.sort()
    # dictionary with counter per class
    gt_counter_per_class = {}
    counter_images_per_class = {}

    gt_files = []
    for txt_file in ground_truth_files_list:
        # print(txt_file)
        file_id = txt_file.split(".txt", 1)[0]
        file_id = os.path.basename(os.path.normpath(file_id))
        # check if there is a correspondent detection-results file
        temp_path = os.path.join(DR_PATH, (file_id + ".txt"))
        if not os.path.exists(temp_path):
            error_msg = "Error. File not found: {}\n".format(temp_path)
            error_msg += "(You can avoid this error message by running extra/intersect-gt-and-dr.py)"
            error(error_msg)
        lines_list = file_lines_to_list(txt_file)
        # create ground-truth dictionary
        bounding_boxes = []
        is_difficult = False
        already_seen_classes = []
        for line in lines_list:
            try:
                if "difficult" in line:
                    class_name, left, top, right, bottom, _difficult = line.split()
                    is_difficult = True
                else:
                    class_name, left, top, right, bottom = line.split()
            except ValueError:
                error_msg = "Error: File " + txt_file + " in the wrong format.\n"
                error_msg += " Expected: <class_name> <left> <top> <right> <bottom> ['difficult']\n"
                error_msg += " Received: " + line
                error_msg += "\n\nIf you have a <class_name> with spaces between words you should remove them\n"
                error_msg += "by running the script \"remove_space.py\" or \"rename_class.py\" in the \"extra/\" folder."
                error(error_msg)
            # check if class is in the ignore list, if yes skip
            if class_name in args.ignore:
                continue
            bbox = left + " " + top + " " + right + " " + bottom
            if is_difficult:
                bounding_boxes.append({"class_name": class_name, "bbox": bbox, "used": False, "difficult": True})
                is_difficult = False
            else:
                bounding_boxes.append({"class_name": class_name, "bbox": bbox, "used": False})
                # count that object
                if class_name in gt_counter_per_class:
                    gt_counter_per_class[class_name] += 1
                else:
                    # if class didn't exist yet
                    gt_counter_per_class[class_name] = 1

                if class_name not in already_seen_classes:
                    if class_name in counter_images_per_class:
                        counter_images_per_class[class_name] += 1
                    else:
                        # if class didn't exist yet
                        counter_images_per_class[class_name] = 1
                    already_seen_classes.append(class_name)

        # dump bounding_boxes into a ".json" file
        new_temp_file = TEMP_FILES_PATH + "/" + file_id + "_ground_truth.json"
        gt_files.append(new_temp_file)
        with open(new_temp_file, 'w') as outfile:
            json.dump(bounding_boxes, outfile)

    gt_classes = list(gt_counter_per_class.keys())
    # let's sort the classes alphabetically
    gt_classes = sorted(gt_classes)
    n_classes = len(gt_classes)

    """
     Check format of the flag --set-class-iou (if used)
        e.g. check if class exists
    """
    if specific_iou_flagged:
        n_args = len(args.set_class_iou)
        error_msg = \
            '\n --set-class-iou [class_1] [IoU_1] [class_2] [IoU_2] [...]'
        if n_args % 2 != 0:
            error('Error, missing arguments. Flag usage:' + error_msg)
        # [class_1] [IoU_1] [class_2] [IoU_2]
        # specific_iou_classes = ['class_1', 'class_2']
        specific_iou_classes = args.set_class_iou[::2]  # even
        # iou_list = ['IoU_1', 'IoU_2']
        iou_list = args.set_class_iou[1::2]  # odd
        if len(specific_iou_classes) != len(iou_list):
            error('Error, missing arguments. Flag usage:' + error_msg)
        for tmp_class in specific_iou_classes:
            if tmp_class not in gt_classes:
                error('Error, unknown class \"' + tmp_class + '\". Flag usage:' + error_msg)
        for num in iou_list:
            if not is_float_between_0_and_1(num):
                error('Error, IoU must be between 0.0 and 1.0. Flag usage:' + error_msg)

    """
     detection-results
         Load each of the detection-results files into a temporary ".json" file.
    """
    # get a list with the detection-results files
    dr_files_list = glob.glob(DR_PATH + '/*.txt')
    dr_files_list.sort()

    for class_index, class_name in enumerate(gt_classes):
        bounding_boxes = []
        for txt_file in dr_files_list:
            # print(txt_file)
            # the first time it checks if all the corresponding ground-truth files exist
            file_id = txt_file.split(".txt", 1)[0]
            file_id = os.path.basename(os.path.normpath(file_id))
            temp_path = os.path.join(GT_PATH, (file_id + ".txt"))
            if class_index == 0:
                if not os.path.exists(temp_path):
                    error_msg = "Error. File not found: {}\n".format(temp_path)
                    error_msg += "(You can avoid this error message by running extra/intersect-gt-and-dr.py)"
                    error(error_msg)
            lines = file_lines_to_list(txt_file)
            for line in lines:
                try:
                    tmp_class_name, confidence, left, top, right, bottom = line.split()
                except ValueError:
                    error_msg = "Error: File " + txt_file + " in the wrong format.\n"
                    error_msg += " Expected: <class_name> <confidence> <left> <top> <right> <bottom>\n"
                    error_msg += " Received: " + line
                    error(error_msg)
                if tmp_class_name == class_name:
                    bbox = left + " " + top + " " + right + " " + bottom
                    bounding_boxes.append({"confidence": confidence, "file_id": file_id, "bbox": bbox})
        # sort detection-results by decreasing confidence
        bounding_boxes.sort(key=lambda x: float(x['confidence']), reverse=True)
        with open(TEMP_FILES_PATH + "/" + class_name + "_dr.json", 'w') as outfile:
            json.dump(bounding_boxes, outfile)

    """
     Calculate the AP for each class
    """
    # open file to store the output
    with open(output_files_path + "/output.txt", 'w') as output_file:
        output_file.write("# AP and precision/recall per class\n")
        count_true_positives = {}
        for class_index, class_name in enumerate(gt_classes):
            count_true_positives[class_name] = 0
            """
             Load detection-results of that class
            """
            dr_file = TEMP_FILES_PATH + "/" + class_name + "_dr.json"
            dr_data = json.load(open(dr_file))

            """
             Assign detection-results to ground-truth objects
            """
            nd = len(dr_data)
            tp = [0] * nd  # creates an array of zeros of size nd
            fp = [0] * nd
            for idx, detection in enumerate(dr_data):
                file_id = detection["file_id"]
                if show_animation:
                    # find ground truth image
                    ground_truth_img = glob.glob1(IMG_PATH, file_id + ".*")
                    # tifCounter = len(glob.glob1(myPath,"*.tif"))
                    if len(ground_truth_img) == 0:
                        error("Error. Image not found with id: " + file_id)
                    elif len(ground_truth_img) > 1:
                        error("Error. Multiple image with id: " + file_id)
                    else:  # found image
                        # print(IMG_PATH + "/" + ground_truth_img[0])
                        # Load image
                        img = cv2.imread(IMG_PATH + "/" + ground_truth_img[0])
                        # load image with draws of multiple detections
                        img_cumulative_path = output_files_path + "/images/" + ground_truth_img[0]
                        if os.path.isfile(img_cumulative_path):
                            img_cumulative = cv2.imread(img_cumulative_path)
                        else:
                            img_cumulative = img.copy()
                        # Add bottom border to image
                        bottom_border = 60
                        BLACK = [0, 0, 0]
                        img = cv2.copyMakeBorder(img, 0, bottom_border, 0, 0, cv2.BORDER_CONSTANT, value=BLACK)
                # assign detection-results to ground truth object if any
                # open ground-truth with that file_id
                gt_file = TEMP_FILES_PATH + "/" + file_id + "_ground_truth.json"
                ground_truth_data = json.load(open(gt_file))
                ovmax = -1
                gt_match = -1
                # load detected object bounding-box
                bb = [float(x) for x in detection["bbox"].split()]
                for obj in ground_truth_data:
                    # look for a class_name match
                    if obj["class_name"] == class_name:
                        bbgt = [float(x) for x in obj["bbox"].split()]
                        bi = [max(bb[0], bbgt[0]), max(bb[1], bbgt[1]), min(bb[2], bbgt[2]), min(bb[3], bbgt[3])]
                        iw = bi[2] - bi[0] + 1
                        ih = bi[3] - bi[1] + 1
                        if iw > 0 and ih > 0:
                            # compute overlap (IoU) = area of intersection / area of union
                            ua = (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1) + (bbgt[2] - bbgt[0]
                                                                              + 1) * (bbgt[3] - bbgt[1] + 1) - iw * ih
                            ov = iw * ih / ua
                            if ov > ovmax:
                                ovmax = ov
                                gt_match = obj

                # assign detection as true positive/don't care/false positive
                if show_animation:
                    status = "NO MATCH FOUND!"  # status is only used in the animation
                # set minimum overlap
                min_overlap = MINOVERLAP
                if specific_iou_flagged:
                    if class_name in specific_iou_classes:
                        index = specific_iou_classes.index(class_name)
                        min_overlap = float(iou_list[index])
                if ovmax >= min_overlap:
                    if "difficult" not in gt_match:
                        if not bool(gt_match["used"]):
                            # true positive
                            tp[idx] = 1
                            gt_match["used"] = True
                            count_true_positives[class_name] += 1
                            # update the ".json" file
                            with open(gt_file, 'w') as f:
                                f.write(json.dumps(ground_truth_data))
                            if show_animation:
                                status = "MATCH!"
                        else:
                            # false positive (multiple detection)
                            fp[idx] = 1
                            if show_animation:
                                status = "REPEATED MATCH!"
                else:
                    # false positive
                    fp[idx] = 1
                    if ovmax > 0:
                        status = "INSUFFICIENT OVERLAP"

                """
                 Draw image to show animation
                """

                print('===================')
                print(ground_truth_img[0])
                print(ovmax)
                print(status)

                result_data = {'Image_filename': [ground_truth_img[0]], 'IoU': [ovmax], 'Status': [status]}
                df = pd.DataFrame(result_data, columns=['Image_filename', 'IoU', 'Status'])
                df_save_path = f'./{class_name}.csv'
                if not os.path.exists(df_save_path):
                    df.to_csv(df_save_path, index=False, mode='w', encoding='euc-kr')
                else:
                    df.to_csv(df_save_path, index=False, mode='a', encoding='euc-kr', header=False)

def backup(src_folder, backup_files, backup_folder):
    # non-intersection files (txt format) will be moved to a backup folder
    if not backup_files:
        print('No backup required for', src_folder)
        return
    os.chdir(src_folder)
    ## create the backup dir if it doesn't exist already
    if not os.path.exists(backup_folder):
        os.makedirs(backup_folder)
    for file in backup_files:
        os.rename(file, backup_folder + '/' + file)

def intersect_gt_and_dr(DIR):
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    GT_PATH = os.path.join(DIR['ROOT_DIR'], 'labels')
    DR_PATH = os.path.join(DIR['ROOT_DIR'], 'results')

    backup_folder = 'backup_no_matches_found'  # must end without slash

    os.chdir(GT_PATH)
    gt_files = glob.glob('*.txt')
    if len(gt_files) == 0:
        print("Error: no .txt files found in", GT_PATH)
        sys.exit()
    os.chdir(DR_PATH)
    dr_files = glob.glob('*.txt')
    if len(dr_files) == 0:
        print("Error: no .txt files found in", DR_PATH)
        sys.exit()

    gt_files = set(gt_files)
    dr_files = set(dr_files)
    print('total ground-truth files:', len(gt_files))
    print('total detection-results files:', len(dr_files))
    print()

    gt_backup = gt_files - dr_files
    dr_backup = dr_files - gt_files

    backup(GT_PATH, gt_backup, backup_folder)
    backup(DR_PATH, dr_backup, backup_folder)
    if gt_backup:
        print('total ground-truth backup files:', len(gt_backup))
    if dr_backup:
        print('total detection-results backup files:', len(dr_backup))

    intersection = gt_files & dr_files
    print('total intersected files:', len(intersection))
    print("Intersection completed!")