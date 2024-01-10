import numpy as np
import cv2
from collections import deque
import GenerareMasca
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
from car import Car
import time

listaPuncte = list()
numarMasti = 0
sp = [12, 13, 14, 15]

def afisare(lista_imagini, text_imagini, cadre_pe_linie=2):
    nr_cadre = len(lista_imagini)
    nr_linii = int(np.ceil(nr_cadre / cadre_pe_linie))
    nr_coloane = cadre_pe_linie
    randuri_poze = []
    *dimensiuni_poza, = lista_imagini[0].shape
    latime = dimensiuni_poza[0]
    inaltime = dimensiuni_poza[1]
    for i in range(nr_linii):
        linie_poze = []
        for j in range(nr_coloane):
            index_lista = i * nr_coloane + j
            if index_lista < nr_cadre:
                cv2.putText(lista_imagini[index_lista], text_imagini[index_lista], (5, 15), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255, 255, 255), 1, cv2.LINE_AA)
                if len(lista_imagini[index_lista].shape) < 3:
                    color = np.zeros((int(latime), int(inaltime), 3), np.uint8)
                    cv2.cvtColor(lista_imagini[index_lista], cv2.COLOR_GRAY2RGB, color)
                    linie_poze.append(color)
                else:
                    linie_poze.append(lista_imagini[index_lista])
            else:
                linie_poze.append(np.zeros((int(latime), int(inaltime), 3), np.uint8))
        randuri_poze.append(cv2.hconcat(linie_poze))
    final = cv2.vconcat(randuri_poze)

    cv2.imshow('video', final)


if __name__ == '__main__':
    car = {}
    recorded_cars = {}
    start_p = {}
    frame_buffer = deque(maxlen=1000)
    yolo_buffer = deque(maxlen=1000)
    current_frame_index = 0
    pause = False
    cap = cv2.VideoCapture('intersectie.mp4')

    model = YOLO('yolov5n.pt')

    MINAREA = 300
    MAXAREA = 50000
    ratio = 0.6 # resize ratio in order to reduce lag

    read_frame_index = 0
    mascaGenerata = GenerareMasca.generareMascaFisier()  # incarca masca facuta cu scriptul GenerareMasca
    # aducem masca la dimensiunea videoclipului
    mascaGenerata = cv2.resize(mascaGenerata, (0, 0), None, 2/3, 2/3)  # resize imageq
    mascaGenerata = cv2.resize(mascaGenerata, (0, 0), None, ratio, ratio)  # resize imageq
    while True:
        start_time = time.time()

        if not pause:
            ret, frame = cap.read()
            if frame is None and read_frame_index == 0:  # nu a citit frame, probabil path prost de fisier
                print("Nu am putut citi niciun frame!")
                break
            elif frame is None and read_frame_index != 0:  # a ajuns la sfarsit, reseteaza frame-ul
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                read_frame_index = 0
                ret, frame = cap.read()

            frame = cv2.resize(frame, (0, 0), None, ratio, ratio)  # resize image
            resizedWidth, resizedHeight, channels = frame.shape
            image = frame.copy()
            frame_buffer.append(image)

            yolo_buffer.append(frame)
            result = model.track(frame, conf=0.20, persist = True, tracker="botsort.yaml", verbose = False)

            for r in result:
                boxes = r.boxes
                offset = 10

                for box in boxes:
                    b = box.xyxy[0]  # get box coordinates in (left, top, right, bottom) format
                    center_x = (b[0] + b[2]) / 2
                    center_y = (b[1] + b[3]) / 2
                    bottom_y = b[3] - offset
                    center_coordinates = (center_x, bottom_y)
                    id = str(int(box.id.item()))
                    masks = GenerareMasca.listaNumereMasca(mascaGenerata[int(center_y)][int(center_x)])

                    if id not in car:
                        c = Car()

                        c.set_center((int(center_coordinates[0]), int(center_coordinates[1])))
                        c.set_id(id)

                        if c.s_point == None:
                            for p in sp:
                                if p in masks:
                                    start_p[id] = [str(val) for val in masks if val != p]
                                    c.set_s_point(start_p[id][0])

                        # print(f'S_POINT : {c.s_point} of id : {c.id}')
                        # print(f'mask lIST : {c.mask_l} of id : {c.id}')
                        c.inc_mask_l(masks)
                        car[c.id] = c
                    else:

                        car[id].set_center((int(center_coordinates[0]), int(center_coordinates[1])))
                        car[id].inc_mask_l(masks)
                        car[id].set_last_frame(read_frame_index)

                    cv2.circle(frame, (int(center_coordinates[0]), int(center_coordinates[1])), 3, (0, 0, 255), -1)

                    textNumere = str(GenerareMasca.listaNumereMasca(mascaGenerata[int(center_y)][int(center_x)]))
                    cv2.putText(frame, textNumere, (int(center_coordinates[0]), int(center_coordinates[1])),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                    text = "ID:"+str(int(box.id.item()))
                    cv2.putText(frame, text, (int(center_coordinates[0]) - 40, int(center_coordinates[1])),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)



        key = cv2.waitKey(10)
        if key == ord('q'):  # Press 'q' to exit
            break
        elif key == ord(' '):  # Press space to toggle pause
            pause = not pause
            current_frame_index = len(frame_buffer) - 1
        elif key == ord('a'):  # Press '<' to move backward
            if pause:
                current_frame_index -= 1
        elif key == ord('d'):  # Press '>' to move forward
            if pause and current_frame_index + 1 < len(frame_buffer) - 1:
                current_frame_index += 1

        # inceput afisare
        #mascaGenerataNoText = mascaGenerata.copy()
        imagini = [frame, yolo_buffer[-1], frame_buffer[-1],mascaGenerata]  # lista cu imagini de afisat
        imagini2 = [frame, yolo_buffer[current_frame_index], frame_buffer[current_frame_index],mascaGenerata]  # lista cu imagini de afisat
        texte = ["frame "+str(read_frame_index), "f", "i", "m"]  # lista cu numele fiecarei imagini
        numar_de_imagini_pe_linie = 2
        if not pause:
            afisare(imagini, texte, numar_de_imagini_pe_linie)
        else:
            afisare(imagini2, texte, numar_de_imagini_pe_linie)

        elapsed_time = time.time() - start_time
        fps = round(1/round(elapsed_time, 3), 0)

        read_frame_index += 1

        #if read_frame_index % (fps * 10) == 0: #  daca au trecut 10 sec
        cars_added = False
        for key, val in car.items():
            if read_frame_index - val.last_frame > 100: #  daca de la ultimul frame inregistrat au trecut alte 100 de frame-uri sterge elementul
                recorded_cars[key] = val
                cars_added = True
        if cars_added:
            for key in recorded_cars:
                if key in car:
                    del car[key]

            for key in recorded_cars:
                print(f'\n------------------------\n'
                      f'CAR ID : {recorded_cars[key].id}\n'
                      f'START POINT : {recorded_cars[key].s_point}\n'
                      f'LAST FRAME : {recorded_cars[key].last_frame}\n'
                      f'FRAME ON EACH MASK : {recorded_cars[key].mask_l}\n'
                      f'--------------------------\n')
        # print(f"Frames per second (FPS): {elapsed_time}")
