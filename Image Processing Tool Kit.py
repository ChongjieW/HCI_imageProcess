import tkinter as tk  # 导入 tkinter 库用于创建 GUI 界面
from tkinter import filedialog  # 导入文件对话框，用于打开图像文件
import cv2  # 导入 OpenCV 库，用于图像处理
import numpy as np  # 导入 NumPy 库，用于数值操作
from PIL import Image, ImageTk  # 导入 PIL 库，用于图像显示
from sklearn.neighbors import KNeighborsClassifier  # 导入 K 最近邻分类器
from sklearn.model_selection import train_test_split  # 导入数据集分割工具
from sklearn.metrics import accuracy_score  # 导入准确度评估工具
import tkinter.messagebox  # 导入消息框，用于显示错误信息
from skimage import exposure

# 为帧声明全局变量
global left_frame  # 左侧帧
global right_frame  # 右侧帧
global image  # 存储图像的全局变量
global canvas  # 画布，用于在界面上显示图像
global original_image_displayed  # 存储原始图像的全局变量
global filtered_image_displayed  # 存储处理后图像的全局变量

# 创建一个固定大小的画布
canvas_width = 1600
canvas_height = 600

# 定义一个函数，用于应用图像滤镜
def apply_filter(filter_type):
    global image  # 使用全局的 image 变量
    img_copy = image.copy()  # 创建 image 的副本以避免修改原始图像
    if filter_type == "Blur":  # 模糊滤镜
        kernel_size = (5, 5)  # 定义模糊的内核大小
        filtered_image = cv2.blur(img_copy, kernel_size)
    elif filter_type == "Sharpen":  # 锐化滤镜
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])  # 定义锐化的内核
        filtered_image = cv2.filter2D(img_copy, -1, kernel)
    elif filter_type == "Edge Detection":  # 边缘检测
        if image is not None:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 转换为灰度图像
            # 使用 Sobel 算子进行边缘检测
            sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=5)
            sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=5)
            magnitude = cv2.magnitude(sobel_x, sobel_y)
            # 将幅度图像转换为 CV_8U 以便显示
            magnitude = cv2.convertScaleAbs(magnitude)
            # 在界面上显示检测到的边缘
            display_filtered_image(magnitude)
        else:
            print("No image to perform edge detection. Please open an image first.")
    elif filter_type == "Image Enhancement":
        if image is not None:
        # Convert the input image to LAB color space
            lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Split the LAB image into L, A, and B channels
            l_channel, a_channel, b_channel = cv2.split(lab_image)
        
        # Apply CLAHE to the L channel (lightness component)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced_l_channel = clahe.apply(l_channel)
        
        # Merge the enhanced L channel with the original A and B channels
            enhanced_lab_image = cv2.merge((enhanced_l_channel, a_channel, b_channel))
        
        # Convert the LAB image back to BGR format
            enhanced_bgr_image = cv2.cvtColor(enhanced_lab_image, cv2.COLOR_LAB2BGR)
        
        # Display the enhanced image
            display_filtered_image(enhanced_bgr_image)

        else:
            print("No image to enhance. Please open an image first.")
    elif filter_type == "Image Segmentation":  # 图像分割
        if image is not None:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 转换为灰度图像
            _, segmented_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            display_filtered_image(segmented_image)
        else:
            print("No image to perform segmentation. Please open an image first.")
    elif filter_type == "Object Recognition":  # 对象识别
        # 使用 Haar 级联分类器进行人脸识别
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 0, 255), 2)  # 在图像上绘制矩形框
        filtered_image = img_copy
    # 在界面上显示处理后的图像
    display_filtered_image(filtered_image)

# 定义一个函数，用于应用图像扭曲
def apply_distortion(distortion_type):
    global image
    global distorted_image

    if distortion_type == "Translation":  # 平移
        translation_matrix = np.float32([[1, 0, 50], [0, 1, 50]])
        distorted_image = cv2.warpAffine(image, translation_matrix, (image.shape[1], image.shape[0]))
    elif distortion_type == "Rotation":  # 旋转
        rotation_matrix = cv2.getRotationMatrix2D((image.shape[1] / 2, image.shape[0] / 2), 45, 1)
        distorted_image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))
    elif distortion_type == "Scaling":  # 缩放
        def set_scale_factor():
            try:
                factor = float(scale_factor_entry.get())
                new_width = int(image.shape[1] * factor)
                new_height = int(image.shape[0] * factor)
                distorted_image = cv2.resize(image, (new_width, new_height))
                display_filtered_image(distorted_image)
                scale_window.destroy()
            except ValueError:
                tk.messagebox.showerror("Error", "Please enter a valid number!")

        scale_window = tk.Toplevel(window)
        scale_window.title("Enter Scaling Factor")

        tk.Label(scale_window, text="Enter scaling factor (e.g., 2 for double size, 0.5 for half size):").pack(pady=10)
        scale_factor_entry = tk.Entry(scale_window)
        scale_factor_entry.pack(pady=10)

        tk.Button(scale_window, text="OK", command=set_scale_factor).pack(pady=10)
    elif distortion_type == "Shear":  # 切变
        shear_matrix = np.float32([[1, 0.3, 0], [0.3, 1, 0]])
        distorted_image = cv2.warpAffine(image, shear_matrix, (image.shape[1], image.shape[0]))
    else:
        distorted_image = image

    display_filtered_image(distorted_image)

# 定义一个函数，用于在界面上显示原始图像
def display_original_image(img):
    global canvas
    global original_image_displayed
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 将图像从 BGR 转换为 RGB 格式
    img = Image.fromarray(img)  # 将 NumPy 数组转换为 PIL 图像
    img = img.resize((canvas_width // 2, canvas_height), Image.ANTIALIAS)  # 调整图像大小
    img = ImageTk.PhotoImage(image=img)  # 创建 Tkinter 图像对象
    canvas.create_image(0, 0, anchor=tk.NW, image=img)  # 在画布上显示原始图像
    original_image_displayed = img  # 保存为全局变量以防止被垃圾回收

# 定义一个函数，用于在界面上显示处理后的图像
def display_filtered_image(img):
    global canvas
    global filtered_image_displayed

    if img.shape[1] > canvas_width // 2 or img.shape[0] > canvas_height:
        img = cv2.resize(img, (canvas_width // 2, canvas_height))  # 调整图像大小以适应画布
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 将图像从 BGR 转换为 RGB 格式
    img = Image.fromarray(img)  # 将 NumPy 数组转换为 PIL 图像
    img = ImageTk.PhotoImage(image=img)  # 创建 Tkinter 图像对象
    canvas.create_image(canvas_width // 2, 0, anchor=tk.NW, image=img)  # 在画布上显示处理后的图像
    filtered_image_displayed = img  # 保存为全局变量以防止被垃圾回收

# 创建主窗口
window = tk.Tk()
window.title("Image Processing Application")  # 设置窗口标题

# 创建菜单栏
menu_bar = tk.Menu(window)
window.config(menu=menu_bar)

# 创建文件菜单
file_menu = tk.Menu(menu_bar, tearoff=0)
menu_bar.add_cascade(label="File", menu=file_menu)

# 添加打开图像选项
def open_image():
    global image
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")])
    if file_path:
        image = cv2.imread(file_path)  # 读取选定的图像文件
        display_original_image(image)  # 在界面上显示原始图像

file_menu.add_command(label="Open Image", command=open_image)

# 创建图像滤波菜单
image_processing_menu = tk.Menu(menu_bar, tearoff=0)
menu_bar.add_cascade(label="Image Filtering", menu=image_processing_menu)

# 添加滤镜类型选项
image_processing_menu.add_command(label="Blur", command=lambda: apply_filter("Blur"))
image_processing_menu.add_command(label="Sharpen", command=lambda: apply_filter("Sharpen"))
image_processing_menu.add_command(label="Edge Detection", command=lambda: apply_filter("Edge Detection"))
image_processing_menu.add_command(label="Image Segmentation", command=lambda: apply_filter("Image Segmentation"))
image_processing_menu.add_command(label="Image Enhancement", command=lambda: apply_filter("Image Enhancement"))

# 添加对象识别选项
image_processing_menu.add_command(label="Object Recognition", command=lambda: apply_filter("Object Recognition"))

# 创建图像扭曲菜单
distortion_menu = tk.Menu(menu_bar, tearoff=0)
menu_bar.add_cascade(label="Image Distortion", menu=distortion_menu)

# 添加图像扭曲类型选项
distortion_menu.add_command(label="Translation", command=lambda: apply_distortion("Translation"))
distortion_menu.add_command(label="Rotation", command=lambda: apply_distortion("Rotation"))
distortion_menu.add_command(label="Scaling", command=lambda: apply_distortion("Scaling"))
distortion_menu.add_command(label="Shear", command=lambda: apply_distortion("Shear"))

# 创建一个固定大小的画布
canvas = tk.Canvas(window, width=canvas_width, height=canvas_height)
canvas.pack()

# 运行主循环
window.mainloop()
