#     fgbg = cv2.createBackgroundSubtractorMOG2()
#
#     ratio = 0.4 # resize ratio in order to reduce lag
#
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
#     kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))

# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # converteste imaginea in grayscale
            # # equalizedHistogram = cv2.equalizeHist(gray)
            # fgmask = fgbg.apply(gray)  # aplica background subtractor pentru a distinge obiectele care se misca
            #
            # # aplicam diferite praguri pt fgmask pentru a incerca sa izolam masinile
            # opening = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel2)  # erosion urmat de dilation
            # mask_buffer.append(opening)
            #
            # # creates contours
            # contours, hierarchy = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            #
            # # use convex hull to create a polygon around contours
            # hull = [cv2.convexHull(c) for c in contours]
            #
            # # draw contours / lines
            # # cv2.drawContours(image, hull, -1, (0, 255, 0), 3)
            # # drawLines(image)
            #
            # cxx = np.zeros(len(contours))
            # cyy = np.zeros(len(contours))
            #
            # for i in range(len(contours)):
            #     if hierarchy[0, i, 3] == -1:  # conturul trebuie sa fie parinte (nu contur din contur)
            #         area = cv2.contourArea(contours[i])
            #         if MINAREA < area < MAXAREA:  # conturul trebuie sa aiba o arie minima si maxima
            #             cnt = contours[i]
            #             M = cv2.moments(cnt)
            #             # stocam x-ul si y-ul conturului
            #             cx = int(M['m10'] / M['m00'])
            #             cy = int(M['m01'] / M['m00'])
            #
            #             x, y, w, h = cv2.boundingRect(cnt)
            #             cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            #             # cv2.putText(image, str(cx) + "," + str(cy), (cx + 10, cy + 10), cv2.FONT_HERSHEY_SIMPLEX, .3, (0, 0, 255), 1)
            #             cv2.drawMarker(image, (cx, cy), (0, 0, 255), cv2.MARKER_STAR, markerSize=5, thickness=1, line_type=cv2.LINE_AA)
            #             textNumere = str(GenerareMasca.listaNumereMasca(mascaGenerata[cy][cx]))
            #             cv2.putText(image, textNumere, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            #             cxx[i] = cx
            #             cyy[i] = cy
            #
            # cxx = cxx[cxx != 0]  # creaza un nou array selectand elementele care satisfac conditia
            # cyy = cyy[cyy != 0]  # practic scapa de elementele cu valoare 0 , raman doar centrele care
            # # raman dupa conditiile de mai sus
            #
            # if len(cxx):  # exista masini detectate in frame
            #     pass