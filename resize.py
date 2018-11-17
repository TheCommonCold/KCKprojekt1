import cv2

for scale in [5.0]:
    for name in ["21","22","23","24","25"]:
        filename = name+".jpg"
        W = 1000.
        oriimg = cv2.imread(filename)
        img = cv2.resize(oriimg, None, fx=1/scale, fy=1/scale)
        cv2.imwrite(name+"-"+str(int(scale))+".jpg", img)
