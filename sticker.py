import cv2
import math
import numpy as np

def get_sticker_img(background_file, sticker_file):
    # 필요한 라이브러리 가져오기
    # 얼굴, 코, 입을 감지하기 위해 Haar cascade 분류기 로드
    face_cascade = cv2.CascadeClassifier('haarcascade/haarcascade_frontalface_alt.xml')
    nose_cascade = cv2.CascadeClassifier('haarcascade/Nariz.xml')
    mouth_cascade = cv2.CascadeClassifier('haarcascade/Mouth.xml')


    # 배경 이미지와 스티커 이미지 읽기
    img = cv2.imread(background_file)  # background img
    sticker_img = cv2.imread(sticker_file)  # 스티커 이미지
    (h, w) = img.shape[:2]  # 배경 이미지의 크기 가져오기

    # 가로와 세로 중 큰 쪽을 720 픽셀로 맞추면서 비율 유지
    if w > h:
        x = 720
        y = int((x * h) / w)
        dim = (x, y)
    else:
        y = 720
        x = int((y * w) / h)
        dim = (x, y)

    resized_img = cv2.resize(img, dim, interpolation=cv2.INTER_LINEAR)  # 배경 이미지 크기 조정
    img_gray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)  # 크기 조정된 이미지를 회색으로 변환하여 인식을 더 잘하기 위함

    # 회색 이미지에서 얼굴을 인식하는데 사용된 Haar cascade 분류기를 이용하여 얼굴을 검출
    faces = face_cascade.detectMultiScale(img_gray, 1.05, 2)  # 얼굴 인식

    for (fx, fy, fw, fh) in faces:
        # 검출된 영역을 약간 확장하여 스티커가 얼굴 영역을 벗어나지 않도록 함
        fx = fx - int(0.1 * fw)
        fy = fy - int(0.1 * fh)
        fw = fw + int(0.2 * fw)
        fh = fh + int(0.26 * fh)

        face = resized_img[fy: fy + fh, fx: fx + fw]  # 크기 조정된 이미지에서 얼굴 영역 추출
        face_gray = img_gray[fy: fy + fh, fx: fx + fw]  # 회색 이미지에서 얼굴 영역 추출

        # 얼굴 영역에서 코와 입을 감지
        nose = nose_cascade.detectMultiScale(face_gray, 1.18, 2)
        mouth = mouth_cascade.detectMultiScale(face_gray, 1.2, 2)

        # 코나 입이 감지되지 않았을 경우 건너뜀
        if not len(nose) or not len(mouth):
            continue

        # 아래 주석을 해제하면 검출된 얼굴, 코, 입 주위에 사각형 그리기(검출된 얼굴 주위에 사각형 그리기)
        # cv2.rectangle(resized_img, (fx, fy), (fx + fw, fy + fh), (255, 0, 0), 2)

        # AI가 인식한 코 후보들 중 얼굴 한가운데에 있는 후보를 코로 선정
        print('nose : ' + str(len(nose)) + ' results found')
        min_distance = 10000
        nx, ny, nw, nh = (0, 0, 0, 0)
        for (_nx, _ny, _nw, _nh) in nose:
            center_nx = int(_nx + _nw / 2)
            center_ny = int(_ny + _nh / 2)
            center_fx = int(fw / 2)
            center_fy = int(fh / 2)
            distance = math.sqrt(math.pow(center_nx - center_fx, 2) + math.pow(center_ny - center_fy, 2))
            if distance < min_distance:
                nx, ny, nw, nh = (_nx, _ny, _nw, _nh)
                min_distance = distance
        center_nose = (int(nx + nw / 2), int(ny + nh / 2))

        # 코 주위에 사각형 그리기
        # cv2.rectangle(face, (nx, ny), (nx + nw, ny + nh), (0, 0, 255), 2)
        # 코 중심에 원 그리기
        # cv2.circle(face,(center_nose[0], center_nose[1]), 2, (0, 0, 255), 2)

        # 입 후보군 중에서 얼굴 제일 아래에 있는 것을 입으로 선택
        # AI가 인식한 입 후보들 중 얼굴 제일 아래에 있는 후보를 입으로 선정
        print('mouth : ' + str(len(mouth)) + ' results found')
        max_my = 0
        mx, my, mw, mh = (0, 0, 0, 0)
        for (_mx, _my, _mw, _mh) in mouth:
            if _my > max_my:
                mx, my, mw, mh = (_mx, _my, _mw, _mh)
                max_my = _my
        center_mouth = (int(mx + mw / 2), int(my + mh / 2))

        # 입 주위에 사각형 그리기
        # cv2.rectangle(face, (mx, my), (mx + mw, my + mh), (0, 0, 0), 2)
        # 입 중심에 원 그리기
        # cv2.circle(face,(center_mouth[0], center_mouth[1]), 2, (0, 0, 0), 2)

        # 스티커 위치 선정
        (m_h, m_w, m_c) = sticker_img.shape
        center_sticker = (int((center_nose[0] + center_mouth[0]) / 2),
                          int((center_nose[1] + center_mouth[1]) / 2 * 1.1))  # 코와 입 중간의 약간 아래쪽
        center_distance = int(math.sqrt(
            math.pow(center_nose[0] - center_mouth[0], 2) + math.pow(center_nose[1] - center_mouth[1],
                                                                     2)))  # 코와 입 사이의 길이
        center_angle = np.rad2deg(math.asin(abs(center_mouth[0] - center_nose[0]) / center_distance))  # 코와 입이 이루는 각
        print("distance : " + str(center_distance))
        print("angle : " + str(center_angle))

        # 스티커 크기 설정
        resized_sticker_img = cv2.resize(sticker_img, (int(3.5 * center_distance), int(2.3 * center_distance)))
        height, width = resized_sticker_img.shape[:2]
        print((center_nose[1], center_mouth[1]))

        # 스티커 회전
        if center_nose[0] <= center_mouth[0]:
            M = cv2.getRotationMatrix2D((width / 2, height / 2), center_angle, 1)
        elif center_nose[0] > center_mouth[0]:
            M = cv2.getRotationMatrix2D((width / 2, height / 2), 360 - center_angle, 1)
        rotated_sticker_img = cv2.warpAffine(resized_sticker_img, M, (width, height))

        # 회전된 스티커 이미지에서 마스크 부분만 추출 - 비트맵 효과
        # sticker : 배경 이미지 위에 씌우는 이미지
        gray_sticker = cv2.cvtColor(rotated_sticker_img, cv2.COLOR_BGR2GRAY)
        _, STICKER_inv = cv2.threshold(gray_sticker, 10, 255, cv2.THRESH_BINARY_INV)  # 색상 값이 10 이상인 부분들은 255로 변경

        background_height, background_width, _ = face.shape
        sticker_height, sticker_width, _ = rotated_sticker_img.shape

        # 스티커 중심 좌표 저장
        sticker_y = center_sticker[0] - int(sticker_width / 2)
        sticker_x = center_sticker[1] - int(sticker_height / 2)

        # 스티커를 적용할 영역 추출
        #         roi = face[sticker_x: sticker_x+sticker_height, sticker_y: sticker_y+sticker_width]
        roi = face[0:sticker_height, 0:sticker_width]

        # 스티커 합성 (add)
        try:
            # roi_sticker = cv2.add(rotated_sticker_img, roi, mask=STICKER_inv)
            roi_sticker = cv2.add(roi, rotated_sticker_img, mask=STICKER_inv)
            result = cv2.add(roi_sticker, rotated_sticker_img)
            np.copyto(roi, result)
        except Exception as e:
            print(f'Exception : {e}\n\n')
            return "roierror"

    # 최종 이미지 저장
    result_url = "static/result_image/" + background_file.split('/')[-1]
    cv2.imwrite(result_url, resized_img)
    cv2.destroyAllWindows()
    # print("---------------")
    return "done"



