import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import Image, ImageTk
import numpy as np
from scipy.signal import convolve2d

# Define Filters
# Edge Detection
kernel_edge = np.array([[-1, -1, -1],
                        [-1, 8, -1],
                        [-1, -1, -1]])


# Sharpen
kernel_sharpen = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])

# Box Blur
kernel_bblur = (1 / 9.0) * np.array([[1., 1., 1.],
                                     [1., 1., 1.],
                                     [1., 1., 1.]])

ori_photo_x, ori_photo_y = 50, 220
new_photo_x, new_photo_y = 680, 220
flag = 0    # 判断有无读取图片


def open_file():
    global ori_img,  ori_photo, new_photo
    file_path = filedialog.askopenfilename()
    if file_path:
        with (open(file_path, 'r') as file):
            img = Image.open(file_path)
            # 调整图片尺寸
            img_x, img_y = img.size
            x = 500
            y = int(x*img_y/img_x)
            img = img.resize((x, y))
            ori_photo = ImageTk.PhotoImage(img)
            ori_img = img
            # ori_canvas = tk.Canvas(root_window, width=x, height=y)
            # img_tag = ori_canvas.create_image(0, 0, anchor='nw', image=ori_photo)
            # ori_canvas.grid(row=0, column=0, columnspan=5)
            global ori_lab, flag
            if flag:
                ori_lab.destroy()
            ori_lab = tk.Label(root_window, image=ori_photo)
            ori_lab.image = ori_photo
            ori_lab.place(x=ori_photo_x, y=ori_photo_y)
            flag = 1


def save_file():
    global ori_img, ori_photo, new_photo
    file_path = filedialog.asksaveasfilename(defaultextension='.png')
    if file_path:
        new_img.save(file_path)


def blur():
    global ori_img, new_img, new_photo
    img = ori_img.copy()
    # new_img = ndimage.gaussian_filter(img, sigma=1)

    # PIL转array
    img = np.asarray(img)
    # 计算卷积和
    transformed_channels = []
    for i in range(3):
        conv_img = convolve2d(img[:,:,i], kernel_bblur, 'valid')
        transformed_channels.append(abs(conv_img))

    new_img = np.dstack(transformed_channels)
    # 限制rgb在0-255范围内
    new_img = np.clip(new_img, 0, 255).astype(np.uint8)

    # array转image
    new_img = Image.fromarray(new_img)
    new_photo = ImageTk.PhotoImage(new_img)
    # 绘图
    new_lab = tk.Label(root_window, image=new_photo)
    new_lab.image = new_photo
    new_lab.place(x=new_photo_x, y=new_photo_y)


def sharpen():
    global ori_img, new_img, new_photo
    img = ori_img.copy()
    # blur1 = ndimage.gaussian_filter(img, sigma=1)
    img = np.asarray(img)
    # new_img = img + 2*(img-blur1)

    transformed_channels = []
    for i in range(3):
        conv_img = convolve2d(img[:, :, i], kernel_sharpen, 'valid')
        transformed_channels.append(abs(conv_img))

    new_img = np.dstack(transformed_channels)
    new_img = np.clip(new_img, 0, 255).astype(np.uint8)

    new_img = Image.fromarray(new_img)
    new_photo = ImageTk.PhotoImage(new_img)
    new_lab = tk.Label(root_window, image=new_photo)
    new_lab.image = new_photo
    new_lab.place(x=new_photo_x, y=new_photo_y)


def edge_detection():
    global ori_img, new_img, new_photo
    img = ori_img.copy()
    img = np.asarray(img)
    transformed_channels = []
    for i in range(3):
        conv_img = convolve2d(img[:, :, i], kernel_edge, 'valid')
        transformed_channels.append(abs(conv_img))

    new_img = np.dstack(transformed_channels)
    new_img = np.clip(new_img, 0, 255).astype(np.uint8)
    new_img = Image.fromarray(new_img)
    new_photo = ImageTk.PhotoImage(new_img)
    new_lab = tk.Label(root_window, image=new_photo)
    new_lab.image = new_photo
    new_lab.place(x=new_photo_x, y=new_photo_y)


def set_bright():
    def show_bright(ev=None):
        global ori_img, ori_photo, new_img, new_photo, new_lab
        img0 = ori_img.copy()
        img0 = np.asarray(img0)
        img1 = img0[:, :, 0:3].copy()       # RGBA转RGB
        new_img = img0.copy()
        img1 = img1 + 255 * scale.get()
        new_img[:, :, 0:3] = np.clip(img1, 0, 255)
        new_img = Image.fromarray(new_img)
        new_photo = ImageTk.PhotoImage(new_img)
        new_lab = tk.Label(root_window, image=new_photo)
        new_lab.image = new_photo
        new_lab.place(x=new_photo_x, y=new_photo_y)

    top = tk.Tk()
    top.geometry('250x150')
    top.title('Set Brightness')
    scale = tk.Scale(top, from_=-1, to=1, resolution=0.1, orient=tk.HORIZONTAL, command=show_bright)
    scale.set(0)
    scale.pack()


def set_contrast():
    global ori_img, ori_photo, new_img, new_photo
    def show_contrast(ev=None):
        img0 = ori_img.copy()
        img0 = np.asarray(img0)
        Threshold = 127 # 平均亮度，取127为近似值
        img1 = img0[:, :, 0:3].copy()
        new_img = img0.copy()
        img1 = img1 + (img1 - Threshold)*scale.get()
        new_img[:, :, 0:3] = np.clip(img1, 0, 255)
        new_img = np.uint8(np.array(new_img, dtype=int))
        new_img = Image.fromarray(new_img)
        new_photo = ImageTk.PhotoImage(new_img)
        new_lab = tk.Label(root_window, image=new_photo)
        new_lab.image = new_photo
        new_lab.place(x=new_photo_x, y=new_photo_y)

    top = tk.Tk()
    top.geometry('250x150')
    top.title('Set Contrast')
    scale = tk.Scale(top, from_=-2, to=2, resolution=0.1, orient=tk.HORIZONTAL, command=show_contrast)
    scale.set(0)
    scale.pack()


def set_saturation():
    global ori_img, ori_photo, new_img, new_photo
    def show_saturation(ev=None):
        img = ori_img.copy()
        img = np.asarray(img)
        img1 = img[:, :, 0:3].copy()
        new_img = img.copy()
        rate = scale.get()
        M = np.float32([
            [1 + 2 * rate, 1 - rate, 1 - rate],
            [1 - rate, 1 + 2 * rate, 1 - rate],
            [1 - rate, 1 - rate, 1 + 2 * rate]
        ])
        shape = img1.shape
        img1 = np.matmul(img1.reshape(-1, 3), M).reshape(shape) / 3
        img1 = np.clip(img1, 0, 255).astype(np.uint8)
        new_img[:, :, 0:3] = img1
        new_img = Image.fromarray(new_img)
        new_photo = ImageTk.PhotoImage(new_img)
        new_lab = tk.Label(root_window, image=new_photo)
        new_lab.image = new_photo
        new_lab.place(x=new_photo_x, y=new_photo_y)

    top = tk.Tk()
    top.geometry('250x150')
    top.title('Set Saturation')
    scale = tk.Scale(top, from_=0, to=10, resolution=0.1, orient=tk.HORIZONTAL, command=show_saturation)
    scale.set(1)
    scale.pack()


# def set_transparency():
#     global ori_img, ori_photo, new_img, new_photo
#     def show_transparency(ev=None):
#         img = ori_img.copy()
#         img = np.asarray(img)
#         img_A = img[:, :, 3].copy()
#         new_img = img.copy()
#         img_A = img_A * scale.get()
#         new_img[:, :, 3] = np.clip(img_A, 0, 255)
#         new_img = Image.fromarray(img)
#         new_photo = ImageTk.PhotoImage(new_img)
#         new_lab = tk.Label(root_window, image=new_photo)
#         new_lab.image = new_photo
#         new_lab.place(x=new_photo_x, y=new_photo_y)
#
#     top = tk.Tk()
#     top.geometry('250x150')
#     top.title('透明度设置')
#     scale = tk.Scale(top, from_=0, to=1, resolution=0.1, orient=tk.HORIZONTAL, command=show_transparency)
#     scale.set(1)
#     scale.pack()


def translation():
    global ori_img, ori_photo, new_img, new_photo
    def show_translation(ev=None):
        img = ori_img.copy()
        width, height = img.size
        # 获取图像通道数
        channel = len(img.split())
        img = np.asarray(img)
        x_delta = int(box_x.get())
        y_delta = int(box_y.get())
        # 构造平移矩阵
        translation_matrix = np.array([[1, 0, 0], [0, 1, 0], [x_delta, y_delta, 1]])
        translated_img = np.zeros([height, width, channel], dtype=np.uint8)
        # 计算每个点的新坐标
        for y in range(height):
            for x in range(width):
                translated_x, translated_y, _ = np.dot([x, y, 1], translation_matrix)
                # 如果平移后的像素点在图像内，则给新图像赋值
                if 0 <= translated_x < width and 0 <= translated_y < height:
                    translated_img[translated_y, translated_x] = img[y, x]

        new_img = Image.fromarray(translated_img)
        new_photo = ImageTk.PhotoImage(new_img)
        new_lab = tk.Label(root_window, image=new_photo)
        new_lab.image = new_photo
        new_lab.place(x=new_photo_x, y=new_photo_y)

    top = tk.Tk()
    top.geometry('250x100')
    top.title('Set Translation')
    box_x = tk.Spinbox(top, from_=0, to=200, increment=10, width=15, command=show_translation)
    box_y = tk.Spinbox(top, from_=0, to=200, increment=10, width=15, command=show_translation)
    box_x.pack()
    box_y.pack()
    # scale_x = tk.Scale(top, from_=0, to=100, resolution=10, orient=tk.HORIZONTAL, command=show_translation)
    # scale_x.set(0)
    # scale_x.pack()
    # scale_y = tk.Scale(top, from_=0, to=100, resolution=10, orient=tk.HORIZONTAL, command=show_translation)
    # scale_y.set(0)
    # scale_y.pack()


def scaling():
    def show_scaling():
        global ori_img, ori_photo, new_img, new_photo
        img = ori_img.copy()
        width, height = img.size
        channel = len(img.split())
        img = np.asarray(img)
        rate = float(box.get())

        # 计算放缩后的图像大小
        new_width = int(np.round(rate*width))
        new_height = int(np.round(rate*height))
        # 创建新图像
        scaled_image = np.zeros((new_height, new_width, channel), dtype=np.uint8)

        # 后向映射
        for x in range(new_width):
            for y in range(new_height):
                # 计算原图坐标
                ori_x = x/rate
                ori_y = y/rate
                # 双线性内插
                left = int(np.floor(ori_x))
                right = int(np.ceil(ori_x))
                top = int(np.floor(ori_y))
                bottom = int(np.ceil(ori_y))
                a = ori_x - left
                b = ori_y - top
                if left >= 0 and right < width and top >= 0 and bottom < height:
                    scaled_image[y, x] = (1 - a) * (1 - b) * img[top, left] + a * (1 - b) * img[top, right] + \
                                          (1 - a) * b * img[bottom, left] + a * b * img[bottom, right]

        global new_lab
        new_lab.destroy()
        new_img = Image.fromarray(scaled_image)
        new_photo = ImageTk.PhotoImage(new_img)
        new_lab = tk.Label(root_window, image=new_photo)
        new_lab.image = new_photo
        new_lab.place(x=new_photo_x, y=new_photo_y)

    tk1 = tk.Tk()
    tk1.geometry('250x100')
    tk1.title('Set Scale')
    var = tk.StringVar(tk1)
    var.set('1')
    box = tk.Spinbox(tk1, from_=0, to=2, increment=0.1, width=15, textvariable=var, command=show_scaling)
    box.pack()


def rotation():
    def show_rotation():
        global ori_img, ori_photo, new_img, new_photo
        img = ori_img.copy()
        width, height = img.size
        channel = len(img.split())
        img = np.asarray(img)

        # 将角度转换为弧度
        angle = int(box.get())
        radians = np.deg2rad(angle)

        cos_theta = np.around(np.cos(radians), decimals=4)
        sin_theta = np.around(np.sin(radians), decimals=4)

        # 计算旋转后的图像大小
        new_width = int(np.round(width * abs(cos_theta) + height * abs(sin_theta)))
        new_height = int(np.round(height * abs(cos_theta) + width * abs(sin_theta)))
        # print(new_width, new_height)
        # 创建新图像
        rotated_image = np.zeros((new_height, new_width, channel), dtype=np.uint8)
        # rotated_image = np.zeros((new_height, new_width, 4), dtype=np.uint8)

        # 计算旋转中心点
        center_x = width / 2
        center_y = height / 2
        # 将坐标系平移回原来的位置，加上自定义旋转点的偏移量
        x_step = (new_width - width) * (center_x / width) + (center_x - 1)
        y_step = (new_height - height) * (center_y / height) + (center_y - 1)
        # translation_matrix1 = np.array([[1, 0, 0], [0, 1, 0], [-center_x, -center_y, 1]])
        # translation_matrix2 = np.array([[1, 0, 0], [0, 1, 0], [x_step, y_step, 1]])
        # # 计算旋转矩阵
        # rotation_matrix = np.array([[cos_theta, sin_theta, 0], [-sin_theta, cos_theta, 0], [0, 0, 1]])

        # # 前向映射，遍历每个像素并进行变换
        # for y in range(height):
        #     for x in range(width):
        #         # 将坐标系平移至中心点
        #         translated_x, translated_y, _ = np.dot([x, y, 1], translation_matrix1)
        #         # 计算旋转后的坐标
        #         rotated_x, rotated_y, _ = np.dot([translated_x, translated_y, 1], rotation_matrix)
        #         rotated_x, rotated_y, _ = np.dot([rotated_x, rotated_y, 1], translation_matrix2)
        #         # 如果旋转后的坐标在原图像范围内，则将该像素复制到新图像中
        #         if 0 <= rotated_x < new_width and 0 <= rotated_y < new_height:
        #             rotated_image[int(np.round(rotated_y)), int(np.round(rotated_x))] = img[y, x]

        mat1 = np.array([[1, 0, 0], [0, 1, 0], [-x_step, -y_step, 1]])
        mat3 = np.array([[1, 0, 0], [0, 1, 0], [center_x, center_y, 1]])
        rotation_matraix =  np.array([[cos_theta, -sin_theta, 0], [sin_theta, cos_theta, 0], [0, 0, 1]])
        # 后向映射
        for x in range(new_width):
            for y in range(new_height):
                # 计算原图坐标
                ori_x, ori_y, _ = np.dot([x, y, 1], mat1)
                ori_x, ori_y, _ = np.dot([ori_x, ori_y, 1], rotation_matraix)
                ori_x, ori_y, _ = np.dot([ori_x, ori_y, 1], mat3)
                # 双线性内插
                left = int(np.floor(ori_x))
                right = int(np.ceil(ori_x))
                top = int(np.floor(ori_y))
                bottom = int(np.ceil(ori_y))
                a = ori_x - left
                b = ori_y - top
                if left >= 0 and right < width and top >= 0 and bottom < height:
                    rotated_image[y, x] = (1 - a) * (1 - b) * img[top, left] + a * (1 - b) * img[top, right] + \
                                          (1 - a) * b * img[bottom, left] + a * b * img[bottom, right]
                # elif left < 0 <= right < width and top >= 0 and bottom < height:
                #     rotated_image[y, x] = (1 - b) * img[top, right] + b * img[bottom, right]
                # elif 0 <= left < width <= right and 0 <= top and bottom < height:
                #     rotated_image[y, x] = (1 - b) * img[top, left] + b * img[bottom, left]
                # elif left >= 0 and right < width and top < 0 <= bottom < height:
                #     rotated_image[y, x] = (1 - a) * img[bottom, left] + a * img[bottom, right]
                # elif left >= 0 and right < width and 0 <= top < height <= bottom:
                #     rotated_image[y, x] = (1 - a) * img[top, left] + a * img[top, right]

        global new_lab
        new_lab.destroy()
        new_img = Image.fromarray(rotated_image)
        new_photo = ImageTk.PhotoImage(new_img)
        new_lab = tk.Label(root_window, image=new_photo)
        new_lab.image = new_photo
        new_lab.place(x=new_photo_x, y=new_photo_y)

    tk1 = tk.Tk()
    tk1.geometry('250x100')
    tk1.title('Set Angles')
    box = tk.Spinbox(tk1, from_=0, to=360, increment=10, width=15, command=show_rotation)
    box.pack()

def shearing():
    def show_shearing():
        global ori_img, ori_photo, new_img, new_photo
        img = ori_img.copy()
        width, height = img.size
        channel = len(img.split())
        img = np.asarray(img)
        sh_x = float(box_x.get())
        sh_y = float(box_y.get())

        # 计算放缩后的图像大小
        new_width = int(width + abs(sh_x*height))
        new_height = int(height + abs(sh_y*width))
        # 创建新图像
        sheared_image = np.zeros((new_height, new_width, channel), dtype=np.uint8)
        # 切变矩阵
        shear_matrix = np.array([[1/(1-sh_x*sh_y), -sh_x/(1-sh_x*sh_y)], [-sh_y/(1-sh_x*sh_y), 1/(1-sh_x*sh_y)]])
        # 后向映射
        for x in range(new_width):
            for y in range(new_height):
                # 计算原图坐标
                ori_x, ori_y = np.dot(shear_matrix, [x, y])
                # 双线性内插
                left = int(np.floor(ori_x))
                right = int(np.ceil(ori_x))
                top = int(np.floor(ori_y))
                bottom = int(np.ceil(ori_y))
                a = ori_x - left
                b = ori_y - top
                if left >= 0 and right < width and top >= 0 and bottom < height:
                    sheared_image[y, x] = (1 - a) * (1 - b) * img[top, left] + a * (1 - b) * img[top, right] + \
                                          (1 - a) * b * img[bottom, left] + a * b * img[bottom, right]

        global new_lab
        new_lab.destroy()
        new_img = Image.fromarray(sheared_image)
        new_photo = ImageTk.PhotoImage(new_img)
        new_lab = tk.Label(root_window, image=new_photo)
        new_lab.image = new_photo
        new_lab.place(x=new_photo_x, y=new_photo_y)

    tk1 = tk.Tk()
    tk1.geometry('250x100')
    tk1.title('Set shearing')
    box_x = tk.Spinbox(tk1, from_=0, to=1, increment=0.1, width=15, command=show_shearing)
    box_x.pack()
    box_y = tk.Spinbox(tk1, from_=0, to=1, increment=0.1, width=15, command=show_shearing)
    box_y.pack()


if __name__ == '__main__':
    # 创建主窗口
    root_window = tk.Tk()
    # 命名
    root_window.title('A Simple Image Processing Tool Kit')
    # 设置窗口大小:宽x高,注,此处不能为 "*",必须使用 "x"
    root_window.geometry("1280x720")
    # # 窗口不能被拉伸
    # root_window.resizable(0,0)
    # 更改左上角窗口的的icon图标
    # root_window.iconbitmap('C:/Users/Administrator/Desktop/bitbug_favicon.ico')
    # # 设置全局变量
    # original = np.ones([400,400,1],np.uint8)*255
    # save_img = np.ones([400,400,1],np.uint8)*127
    # sav_lab = tk.Label(root_window)

    # 创建主目录菜单（顶级菜单）
    mainmenu = Menu(root_window)
    # 在顶级菜单上新增"文件"菜单的子菜单，同时不添加分割线
    filemenu = Menu(mainmenu, tearoff=False)
    # 新增"文件"菜单的菜单项，并使用 accelerator 设置菜单项的快捷键
    # filemenu.add_command(label="New", accelerator="Ctrl+N")
    filemenu.add_command(label="Open", command=open_file, accelerator="Ctrl+O")
    filemenu.add_command(label="Save", command=save_file, accelerator="Ctrl+S")
    # 添加一条分割线
    filemenu.add_separator()
    filemenu.add_command(label="Exit", command=root_window. quit)
    # 在主目录菜单上新增"文件"选项，并通过menu参数与下拉菜单绑定
    mainmenu.add_cascade(label="File", menu=filemenu)
    # 将主菜单设置在窗口上
    root_window.config(menu=mainmenu)
    # 绑定键盘事件，按下键盘上的相应的键时都会触发执行函数
    # root_window.bind("<Control-n>")
    # root_window. bind("<Control-N>")
    root_window.bind("<Control-o>")
    root_window. bind("<Control-O>")
    root_window. bind("<Control-s>")
    root_window.bind("<Control-S>")

    new_lab = tk.Label(root_window)

    #创建标签
    tk.Label(root_window, text='Original Image').grid(row=2, column=1, sticky='w', padx=100, pady=10)
    tk.Label(root_window, text='Processed Image').grid(row=2, column=4, sticky='n', padx=10, pady=10)

    # 图像模糊
    button_blur = tk.Button(root_window, text='Blur', bg='#87CEEB', width=15, height=3,command=blur)
    button_blur.grid(row=0, column=0, sticky='w', padx=20, pady=10)
    # 图像锐化
    button_sharpen = tk.Button(root_window, text='Sharpen', bg='#87CEEB', width=15, height=3,command=sharpen)
    button_sharpen.grid(row=0, column=1, sticky='n', padx=10, pady=10)
    # 边缘检测
    button_detection = tk.Button(root_window, text='Edge Detection', bg='#87CEEB', width=15, height=3,command=edge_detection)
    button_detection.grid(row=0, column=2, sticky='n', padx=10, pady=10)

    # 平移
    button_translation = tk.Button(root_window, text='Translation', bg='#7CCD7C', width=15, height=3,command=translation)
    button_translation.grid(row=1, column=0, sticky='n', padx=10, pady=10)
    # 放缩
    button_scaling = tk.Button(root_window, text='Scaling', bg='#7CCD7C', width=15, height=3,command=scaling)
    button_scaling.grid(row=1, column=1, sticky='n', padx=10, pady=10)
    # 旋转
    button_rotation = tk.Button(root_window, text='Rotation', bg='#7CCD7C', width=15, height=3,command=rotation)
    button_rotation.grid(row=1, column=2, sticky='n', padx=10, pady=10)
    # 切变
    button_shearing = tk.Button(root_window, text='Shearing', bg='#7CCD7C', width=15, height=3, command=shearing)
    button_shearing.grid(row=1, column=3, sticky='n', padx=10, pady=10)

    # 设置亮度
    button_bright = tk.Button(root_window, text='Set Brightness', bg='#FFD700', width=15, height=3,command=set_bright)
    button_bright.grid(row=0, column=3, sticky='n', padx=100, pady=10)
    # 设置对比度
    button_contrast = tk.Button(root_window, text='Set Contrast', bg='#FFD700', width=15, height=3,command=set_contrast)
    button_contrast.grid(row=0, column=4, sticky='n', padx=10, pady=10)
    # 设置饱和度
    button_saturation = tk.Button(root_window, text='Set Saturation', bg='#FFD700', width=15, height=3,command=set_saturation)
    button_saturation.grid(row=0, column=5, sticky='n', padx=100, pady=10)

    # # 设置透明度
    # button_transparency = tk.Button(root_window, text='Set Transparency', bg='#7CCD7C', width=15, height=3,
    #                               command=set_transparency).grid(row=1, column=4, sticky='n', padx=100, pady=10)

    # ori_frame = tk.Frame()
    # ori_frame.grid(row=5, column=0, columnspan=7)
    # new_frame = tk.Frame()
    # new_frame.grid(row=5, column=9, columnspan=7)
    #
    # ori_canvas = tk.Canvas(ori_frame, bg="#ffd9b3")
    # new_canvas = tk.Canvas(new_frame, bg="#ffd9b3")

    # 开启主循环
    root_window.mainloop()



