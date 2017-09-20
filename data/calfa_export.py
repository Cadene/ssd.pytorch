"""CALFA Dataset
"""

import os
import sys
import argparse

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

sys.path.insert(0, '/home/cadene/Documents/calfa_new')
import calfa.datasets as datasets
import calfa.lib.make_lines as make_lines

parser = argparse.ArgumentParser(description='PyTorch LinesDataset Evaluation')
parser.add_argument('--dir_calfa_in', metavar='DIR', default='/local/cadene/data/calfa/raw/v1',
                    help='path to dataset')
parser.add_argument('--dir_calfa_out', metavar='DIR', default='/local/cadene/data/CalfaDevkit/CalfaV1',
                    help='path to dataset')


def write_main(dir_main, lines, set_name):
    os.system('mkdir -p ' + dir_main)
    path_txt = os.path.join(dir_main, set_name+'.txt')
    with open(path_txt, 'w') as handle:
        for image_id, _ in lines:
            handle.write(str(image_id)+'\n')

def write_img(dir_img, lines):
    os.system('mkdir -p ' + dir_img)
    for image_id, line in lines:
        image = line.get_image()
        path_image = os.path.join(dir_img, str(image_id)+'.jpg')
        image.save(path_image)

def write_anno(dir_anno, lines):
    os.system('mkdir -p ' + dir_anno)
    for image_id, line in lines:
        path_anno = os.path.join(dir_anno, str(image_id)+'.xml')
        image = line.get_image()
        line_coord = line.get_coord()
        width, height = image.size

        et_annotation = ET.Element('annotation')
        et_folder = ET.SubElement(et_annotation, 'folder')
        et_folder.text = 'CalfaV1'
        et_filename = ET.SubElement(et_annotation, 'filename')
        et_filename.text = str(image_id)+'.jpg'
        et_size = ET.SubElement(et_annotation, 'size')
        et_width = ET.SubElement(et_size, 'width')
        et_width.text = str(width)
        et_height = ET.SubElement(et_size, 'height')
        et_height.text = str(height)
        et_depth = ET.SubElement(et_size, 'depth')
        et_depth.text = '3'

        for _, char in line.get_chars().items():
            xmin = char.get_coord('x') - line_coord['x']
            ymin = char.get_coord('y') - line_coord['y']
            xmax = xmin + char.get_coord('width')
            ymax = ymin + char.get_coord('height')

            et_object = ET.SubElement(et_annotation, 'object')
            et_name = ET.SubElement(et_object, 'name')
            et_name.text = char.get_class_name()

            et_difficult = ET.SubElement(et_object, 'difficult')
            et_difficult.text = '0'

            et_bnbox = ET.SubElement(et_object, 'bndbox')
            et_xmin = ET.SubElement(et_bnbox, 'xmin')
            et_xmin.text = str(xmin)
            et_ymin = ET.SubElement(et_bnbox, 'ymin')
            et_ymin.text = str(ymin)
            et_xmax = ET.SubElement(et_bnbox, 'xmax')
            et_xmax.text = str(xmax)
            et_ymax = ET.SubElement(et_bnbox, 'ymax')
            et_ymax.text = str(ymax)

        #with open(path_anno, 'w') as handle:
        ET.ElementTree(et_annotation).write(path_anno, encoding="UTF-8")


def main():

    args = parser.parse_args()

    train_library = datasets.factory_library(args.dir_calfa_in, 'train', with_lines=True,
                                             lines_strat=make_lines.strategy_coord)
    val_library = datasets.factory_library(args.dir_calfa_in, 'val', with_lines=True,
                                             lines_strat=make_lines.strategy_coord)
    
    train_list_lines = list(train_library.get_lines().values())
    val_list_lines = list(val_library.get_lines().values())

    image_id = 0
    train_lines = []
    for line in train_list_lines:
        train_lines.append((image_id, line))
        image_id += 1
    val_lines = []
    for line in val_list_lines:
        val_lines.append((image_id, line))
        image_id += 1

    ### ImageSets/Main
    dir_main = os.path.join(args.dir_calfa_out, 'ImageSets', 'Main')
    write_main(dir_main, train_lines, 'train')
    write_main(dir_main, val_lines, 'val')

    ### JPEGImages
    dir_img = os.path.join(args.dir_calfa_out, 'JPEGImages')
    write_img(dir_img, train_lines)
    write_img(dir_img, val_lines)

    ### Annotations
    dir_anno = os.path.join(args.dir_calfa_out, 'Annotations')
    write_anno(dir_anno, train_lines)
    write_anno(dir_anno, val_lines)


if __name__ == '__main__':
    main()


