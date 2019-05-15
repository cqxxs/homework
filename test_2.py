import cv2
import numpy as np
import matplotlib.pyplot as plt



# 1.摄像头访问
cam = cv2.VideoCapture(0)
cv2.namedWindow('disp',cv2.WINDOW_AUTOSIZE)

fps = 30
nframe = fps*10 - 1
success,frame = cam.read()
foucc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('out.avi',foucc,fps,(640,480))

while success and nframe > 0:
    cv2.imshow('disp',frame)
    success,frame = cam.read()
    out.write(frame)
    nframe -= 1
cv2.waitKey(1)
cam.release()
cv2.destroyAllWindows()
exit()
# 2.视频访问
video = cv2.VideoCapture('out.avi')
# 3.将摄像头捕捉图像保存到 cvcourse.avi

# 4.使用傅里叶变换压缩图像。分别保存频谱中心范围的60% 30% 10% 1%的数据，然后逆变换到空域观察
img = cv2.imread('d://img/touxiang.jpg',0)

def fft(n):
    x,y = img.shape
    x,y = x/10,y/10
    w = (10-n)*x/2
    h = (10-n)*y/2
    rows,cols = img.shape
    print(rows,cols)
    crow,ccol = rows/2, cols/2
    crow = int(crow)
    ccol = int(ccol)
    fshift[crow-x:crow+x,cols-y:cols+y]=0
    f_shift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_shift)
    img_back = np.abs(img_back)
    plt.imshow(img_back,cmap='gray')
    plt.show()
import numpy as np
import matplotlib.pyplot as plt
import cv2

# img = cv2.imread('d://img/touxiang.jpg',0)
#
# f = np.fft.fft2(img)
# fshift = np.fft.fftshift(f)
#
#
# sss = 20*np.log(np.abs(fshift))
#
# plt.imshow(sss,cmap='gray')
# plt.show()
# rows,cols = img.shape
# print(rows,cols)
#
# crow,ccol = rows/2, cols/2
# crow = int(crow)
# ccol = int(ccol)
# print(ccol,crow)
# fshift[crow-30:crow+30,cols-30:cols+30]=0
# f_shift = np.fft.ifftshift(fshift)
# img_back = np.fft.ifft2(f_shift)
# img_back = np.abs(img_back)
# plt.imshow(img_back,cmap='gray')
# plt.show()


def fft(n):
    '''
    傅里叶变换，图像操作
    :param n: 保留图片的百分比 如 60 30
    :return: 压缩后的图片
    '''
    img = cv2.imread('d://img/touxiang.jpg', 0)


    x,y = img.shape

    x,y = x/100,y/100
    w = int((100-n)*x/2)
    h = int((100-n)*y/2)
    f = np.fft.fft2(img)

    fshift = np.fft.fftshift(f)
    rows,cols = img.shape
    crow,ccol = rows/2, cols/2
    crow = int(crow)
    ccol = int(ccol)
    fshift[crow-w:crow+w,cols-h:cols+h]=0
    f_shift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_shift)
    img_back = np.abs(img_back)
    return img_back
plt.figure()
plt.subplot(221)
plt.title('保留60')
plt.imshow(fft(60),cmap='gray')
plt.subplot(222)
plt.title('30')
plt.imshow(fft(30),cmap='gray')
plt.subplot(223)
plt.title('10')
plt.imshow(fft(10),cmap='gray')
plt.subplot(224)
plt.title('1')
plt.imshow(fft(1),cmap='gray')
plt.show()
# 图像质量的变化，并作图像记录




# 5.测试光流算法
