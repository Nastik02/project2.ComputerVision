
import numpy as np
import cv2

##########################################################################################################
# Функция находит буквы на выделенном изображении и отмечает их.
def find_text(img):


    filterd_image = cv2.medianBlur(img, 1) #Подбором выбрала медианный блюр, 1 - размер ядра
    # filterd_image  = cv2.GaussianBlur(resized_down,(5,5),0)


    img_grey = cv2.cvtColor(filterd_image, cv2.COLOR_BGR2GRAY)

    thresh = 75

    ret, thresh_img = cv2.threshold(img_grey, thresh, 255, cv2.THRESH_BINARY)

    # нахождение контуров
    contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    img_contours = np.uint8(np.zeros((img.shape[0], img.shape[1])))

    # создание массива контуров с подходящим размером площади

    grouped_contours = []
    max = 1500
    min = 300
    for countour in contours:
        if cv2.contourArea(countour) < max and cv2.contourArea(countour) > min:
            grouped_contours.append(countour)

    cv2.drawContours(img_contours, grouped_contours, -1, (255, 255, 255), 1)

    # определение букв по высоте и ширине контура
    for countour in grouped_contours:
        x, y, w, h = cv2.boundingRect(countour)
        if 20 < h < 50 and 15 < w < 60:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.imshow('origin', img)  # выводим итоговое изображение в окно
    cv2.imshow('res2', img_contours)  # выводим итоговое изображение в окно
    cv2.imwrite('text.jpg', img)


#######################################################################################################
def laplacian_focus_measure(img, threshold=100):
    # Загружаем изображение
    image = img

    # Преобразуем в оттенки серого
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Вычисляем Лапласиан
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    # Проверяем, превышает ли значение порог
    return laplacian_var > threshold
##############################################################################################
# Ниже код для записи платы с камеры, подключенной к компьютеру
"""
cap =cv2.VideoCapture(1,cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Ошибка: Не удалось открыть камеру.")
    exit()


frame_count = 0  # Счетчик кадров

while True:
    # Читаем следующий кадр
    success, frame = cap.read()


    # Если кадр не был прочитан, значит, видео закончилось
    if not success:
        cv2.rectangle(frame, (100, 100), (200, 200), [255, 0, 0], 2)
        print("Кадры прочитаны.")
        break


    if laplacian_focus_measure(frame) and success:
        print(f"Кадр {frame_count} прочитан.")
        focus_image = frame
        break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


    frame_count += 1
"""

focus_image = cv2.imread('gkfnf1.jpg')
down_width = 700
down_height = 500
down_points = (down_width, down_height)
focus_image = cv2.resize(focus_image, down_points, interpolation= cv2.INTER_LINEAR)
#filterd_image  = cv2.medianBlur(focus_image,7)
filterd_image  = cv2.GaussianBlur(focus_image,(5,5),0)
gray = cv2.cvtColor(filterd_image, cv2.COLOR_BGR2GRAY)
tr = 110

ret,thresh = cv2.threshold(gray, tr, 255, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
img_contours = np.uint8(np.zeros((focus_image.shape[0],focus_image.shape[1])))

grouped_contours = []
max=100000
i = 0
sel_countour=None
for countour in contours:
    print(cv2.contourArea(countour))
    if cv2.contourArea(countour)<max and cv2.contourArea(countour)>15000:
        grouped_contours.append(countour)


cv2.drawContours(img_contours, grouped_contours, -1, (255,255,255), 1)
# Вырезание обьектов в отдельные изображения
for c in grouped_contours:
    x, y, w, h = cv2.boundingRect(c)
    cv2.rectangle(focus_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
    start_y = int(y)
    end_y = int(y + h)
    start_x = int(x)
    end_x = int(x + w)
    print(f'центр y: {y}')
    print(f'старт y: {start_y}')
    copy = focus_image
    crop_img = copy[start_y:end_y, start_x:end_x]


for c in grouped_contours:
       # Получаем минимальный охватывающий прямоугольник
       rect = cv2.minAreaRect(c)
       box = cv2.boxPoints(rect)  # Получаем координаты вершин прямоугольника
       box = np.uint8(box)

       # Извлекаем угол наклона
       angle = rect[2]  # Угол наклона

       # Корректируем угол
       if angle < -45:
           angle += 90

       print(f'Угол наклона объекта: {angle}')


h, w = crop_img.shape[:2]
center = (int(w / 2), int(h / 2))
rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.3)
rotated_crop_img = cv2.warpAffine(crop_img, rotation_matrix, (w, h))

find_text(rotated_crop_img)
cv2.imshow('res', focus_image) # выводим итоговое изображение в окно
cv2.imshow("Thresh", thresh)
cv2.imwrite('object.jpg', focus_image)


#cap.release()
cv2.waitKey(0)





