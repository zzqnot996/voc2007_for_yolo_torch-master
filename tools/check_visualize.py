import os
import random
import shutil
from xml.etree.ElementTree import Element, parse
import cv2

'''
用于处理图像和XML标注文件的Python脚本
'''

def mk(path):

    '''
    判断是否存在该文件夹
    '''
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        print('There are %d files in %s' % (len(os.listdir(path)), path))


def checkJpgXml(jpg_dirs,
                xml_dirs,
                empty_dirs,
                intersection_jpg_dir,
                intersection_xml_dir,
                is_move=True):
    """
    jpg_dirs 是图片所在文件夹
    dir2 是标注文件所在文件夹
    dir3 是创建的，如果图片没有对应的xml文件，那就将图片放入dir3
    is_move 是确认是否进行移动，否则只进行打印
    """

    set1 = set()
    set2 = set()

    for i in os.listdir(jpg_dirs):
        set1.add(i.split('.')[0])

    for j in os.listdir(xml_dirs):
        set2.add(j.split('.')[0])

    intersection = set1 & set2

    mk(empty_dirs)
    mk(intersection_jpg_dir)
    mk(intersection_xml_dir)

    jpg_error_set = set1 - intersection
    xml_error_set = set2 - intersection

    print('###########  right jpgs  ###########')
    for _name in intersection:

        path1 = os.path.join(jpg_dirs, _name + '.jpg')
        path2 = os.path.join(intersection_jpg_dir, _name + '.jpg')
        shutil.copy(path1, path2)

        path1 = os.path.join(xml_dirs, _name + '.xml')
        path2 = os.path.join(intersection_xml_dir, _name + '.xml')
        shutil.copy(path1, path2)

    if len(jpg_error_set) > 0 or len(xml_error_set):
        print('There are %d error jpg and %d error xml' %
              (len(jpg_error_set), len(xml_error_set)))

        if is_move:
            print('###########  Error jpgs  ###########')
            for _jpg in jpg_error_set:
                print(_jpg + '.jpg')
                path1 = os.path.join(jpg_dirs, _jpg + '.jpg')
                path2 = os.path.join(empty_dirs, _jpg + '.jpg')
                shutil.move(path1, path2)

            print('###########  Error xmls  ###########')
            for _xml in xml_error_set:
                print(_xml + '.xml')
                path1 = os.path.join(xml_dirs, _xml + '.xml')
                path2 = os.path.join(empty_dirs, _xml + '.xml')
                shutil.move(path1, path2)
            print('==============end==================')
        return False

    else:
        print('所有图片和对应的xml文件都是一一对应的。')
        return True


def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    # # tl是线条的粗细，如果未提供则根据图像大小自动计算
    tl = line_thickness or round(
        0.001 * (img.shape[0] + img.shape[1]) / 2) + 1  # line thickness
    # 如果未提供颜色，则随机生成一个颜色
    color = color or [random.randint(0, 255) for _ in range(3)]
    # 获取矩形框的两个顶点坐标
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    # # 在图像上绘制矩形框
    cv2.rectangle(img, c1, c2, color, thickness=tl)
    # # 如果提供了标签，则在矩形框上添加标签
    if label:
        # tf是字体的粗细
        tf = max(tl - 1, 1)  # font thickness
        # 计算标签的大小
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        # 调整矩形框底部，以适应标签
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        # 在矩形框上绘制填充的矩形，以容纳标签
        cv2.rectangle(img, c1, c2, color, -1)  # filled
        # 在图像上添加标签文本
        cv2.putText(img,
                    label, (c1[0] - 3, c1[1] - 2),
                    0,
                    tl / 3, [225, 255, 255],
                    thickness=tf,
                    lineType=cv2.LINE_AA)


def isotropically_resize_image(img, size, resample=cv2.INTER_AREA):
    # # 获取原始图像的高度和宽度
    h, w = img.shape[:2]
    # 如果宽度大于高度，将高度等比例缩放，使其符合目标大小，并调整宽度
    if w > h:
        h = h * size // w
        w = size
    else:
        # 如果高度大于宽度，将宽度等比例缩放，使其符合目标大小，并调整高度
        w = w * size // h
        h = size
    # 使用cv2的resize函数进行图像缩放，保持宽高比
    resized = cv2.resize(img, (w, h), interpolation=resample)
    return resized


def make_square_image(img):
    '''用于将图像填充为正方形，通过在图像的边缘添加边框'''
    # # 获取原始图像的高度和宽度
    h, w = img.shape[:2]
    # 计算填充后的正方形边长
    size = max(h, w)
    # 计算需要在图像上、下、左、右添加的边框像素数
    t = 0
    b = size - h
    l = 0
    r = size - w
    # 使用cv2的copyMakeBorder函数，在图像的边缘添加边框
    return cv2.copyMakeBorder(img, t, b, l, r, cv2.BORDER_CONSTANT, value=0)


def rescale_img_bbox(xml_path, jpg_path, resizedSize, save_xml_path,
                     save_jpg_path):

    '''将给定的图像和XML标注文件进行缩放，以适应指定的大小，并保存缩放后的图像和XML标注文件'''
    # 创建保存缩放后图像和XML的目录
    mk(save_jpg_path)
    mk(save_xml_path)

    # 获取图像文件名（不包括扩展名）
    fileName = os.path.basename(jpg_path).split('.')[0]

    # 解析XML文件
    dom = parse(xml_path)
    root = dom.getroot()
    print('===:', jpg_path)

    # 读取图像
    img = cv2.imread(jpg_path)
    print(img.shape)
    # 等比例缩放图像并将其调整为正方形
    img = isotropically_resize_image(img, resizedSize)
    img = make_square_image(img)

    # img = cv2.resize(img, (w, h))
    # 获取原始图像的宽度和高度
    ssize = root.find('size')
    w = int(ssize.find('width').text)
    h = int(ssize.find('height').text)

    # 遍历XML文件中的每个对象（边界框）
    for obj in root.iter('object'):
        # get scale   计算缩放比例
        w_scale = resizedSize / w
        h_scale = resizedSize / h

        # get coords 获取边界框的坐标
        tmp_name = obj.find('name').text
        xmlbox = obj.find('bndbox')
        x1, y1 = xmlbox.find('xmin').text, xmlbox.find('ymin').text
        x2, y2 = xmlbox.find('xmax').text, xmlbox.find('ymax').text
        print('IN:', x1, y1, x2, y2)
        x1, y1 = int(x1), int(y1)
        x2, y2 = int(x2), int(y2)

        # rescale 缩放坐标
        x1, x2 = x1 * w_scale, x2 * w_scale
        y1, y2 = y1 * h_scale, y2 * h_scale

        # make new xml 更新XML文件中的边界框坐标
        xmlbox.find('xmin').text = str(int(x1))
        xmlbox.find('ymin').text = str(int(y1))
        xmlbox.find('xmax').text = str(int(x2))
        xmlbox.find('ymax').text = str(int(y2))

        _box = [x1, y1, x2, y2]
        print('out:', _box)
        # 如果需要，可以在图像上绘制边界框
        # plot_one_box(_box, img, label=tmp_name)

    # 更新XML文件中的图像宽度和高度
    ssize = root.find('size')
    ssize.find('width').text = str(resizedSize)
    ssize.find('height').text = str(resizedSize)

    # 保存缩放后的图像和XML文件
    cv2.imwrite(os.path.join(save_jpg_path, fileName + '.jpg'), img)
    dom.write(os.path.join(save_xml_path, fileName + '.xml'),
              xml_declaration=True)



def changeName(xml_fold, origin_name, new_name):
    '''
    处理图像标注

    xml_fold: xml存放文件夹
    origin_name: 原始名字，比如弄错的名字，原先要cow,不小心打成cwo
    new_name: 需要改成的正确的名字，在上个例子中就是cow
    '''
    # 获取指定文件夹中所有文件列表
    files = os.listdir(xml_fold)
    # 记录成功修改的文件数量
    cnt = 0
    # 遍历每个XML文件
    for xmlFile in files:
        # 构建XML文件的完整路径
        file_path = os.path.join(xml_fold, xmlFile)

        # 解析XML文件内容
        dom = parse(file_path)
        root = dom.getroot()

        # 遍历XML文件中的每个目标物体（object节点）
        for obj in root.iter('object'):  # 获取object节点中的name子节点
            # 获取目标物体的当前类别名称
            tmp_name = obj.find('name').text

            # 如果目标物体的类别名称与指定的原始类别名称相同，进行修改
            if tmp_name == origin_name:  # 修改
                obj.find('name').text = new_name
                print('change %s to %s.' % (origin_name, new_name))
                cnt += 1

        # 保存更新后的XML文件
        dom.write(file_path, xml_declaration=True)  #保存到指定文件

    # 打印成功修改的文件数量
    print('有%d个文件被成功修改。' % cnt)


# if __name__ == "__main__":
#     changeName(xml_fold=r"/home/ubuntu/yolov3/voc2007Crack-labels（1）\C00",
#                origin_name='C00',
#                new_name='crack')

if __name__ == '__main__':
    # jpg_path = r"/home/ubuntu/yolov3/voc2007crack (329).jpg"
    # xml_path = r"/home/ubuntu/yolov3/voc2007crack (329).xml"

    jpg_dirs = '/home/ubuntu/yolov3/voc2007/JPEGImages'
    # r"/home/ubuntu/yolov3/voc2007/Annotations/JPEGImages"

    xml_dirs = '/home/ubuntu/yolov3/voc2007/Annotations'

    empty_dirs = r'/home/ubuntu/yolov3/voc2007/empty'

    intersection_jpg_dir = r'/home/ubuntu/yolov3/voc2007/interset_jpg'
    intersection_xml_dir = r'/home/ubuntu/yolov3/voc2007/interset_xml'

    judge = checkJpgXml(jpg_dirs, xml_dirs, empty_dirs, intersection_jpg_dir,
                        intersection_xml_dir)

    save_jpg_path = r'/home/ubuntu/yolov3/voc2007/outjpgs'
    save_xml_path = r'/home/ubuntu/yolov3/voc2007/outxmls'

    resizedSize = 416

    for file in os.listdir(intersection_jpg_dir):
        fileName = file.split('.')[0]
        print(fileName)

        jpg_file_path = os.path.join(intersection_jpg_dir, fileName + '.jpg')
        xml_file_path = os.path.join(intersection_xml_dir, fileName + '.xml')

        rescale_img_bbox(xml_file_path, jpg_file_path, resizedSize,
                         save_xml_path, save_jpg_path)
