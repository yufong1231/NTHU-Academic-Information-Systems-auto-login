from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import cv2

#define how many images do you want do get
images_numbers = 1000

def catch_data():
    #don't show window
    chrome_options = Options()
    chrome_options.add_argument("--headless")

    driver = webdriver.Chrome(chrome_options=chrome_options)
    driver.set_window_size(1200, 900)
    driver.get('https://www.ccxp.nthu.edu.tw/ccxp/INQUIRE/')
    #get the confrim number images
    for count in range(images_numbers):
        driver.save_screenshot('./screenshot.jpg')
        img = cv2.imread('./screenshot.jpg')
        img = img[514:576, 85:285]

        cv2.imwrite("./numbers/" + str(count) + ".jpg", img)
        driver.refresh()

    driver.close()

def split_numbers(label, path):
    print(path)
    img = cv2.imread(path)
    # origin image -> gray -> threshold
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, img_threshold = cv2.threshold(img_gray, 150, 255, cv2.THRESH_BINARY_INV);
    ret, img_output = cv2.threshold(img_gray, 150, 255, cv2.THRESH_BINARY);
    #find contours
    image_contours, contours, hierarchy = cv2.findContours(img_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #error case
    if len(contours) > 6:
        print("Error : detect too many coutours!")
    #two numbers connect case
    elif len(contours) < 6 and len(contours) > 0:
        max_width = 0
        count = 0
        #find max width of all contours
        for i in range(0,len(contours)):
            x, y, w, h = cv2.boundingRect(contours[i])
            if w >= max_width:
                max_width = w
        #split maxwidth contours into two contours
        for i in range(0,len(contours)):
            x, y, w, h = cv2.boundingRect(contours[i])
            if w == max_width:
                #cv2.rectangle(img_output, (x,y), (int(x+w/2), y+h), (0, 255, 0), 1)
                #cv2.rectangle(img_output, (int(x+w/2),y), (x+w, y+h), (0, 255, 0), 1)
                output = cv2.resize(img_output[y:y+h, x:int(x+w/2)], (28,28), interpolation=cv2.INTER_CUBIC)
                cv2.imwrite("./data/" + str(label) + "_" + str(count) + ".jpg", output)
                count += 1
                output = cv2.resize(img_output[y:y+h, int(x+w/2):x+w], (28,28), interpolation=cv2.INTER_CUBIC)
                cv2.imwrite("./data/" + str(label) + "_" + str(count) + ".jpg", output)
                count += 1
            else:
                #cv2.rectangle(img_output, (x,y), (x+w, y+h), (0, 255, 0), 1)
                output = cv2.resize(img_output[y:y+h, x:x+w], (28,28), interpolation=cv2.INTER_CUBIC)
                cv2.imwrite("./data/" + str(label) + "_" + str(count) + ".jpg", output)
                count += 1
    #normal case (can detect accurate six numbers)
    else:
        count = 0
        for i in range(0,len(contours)):
            x, y, w, h = cv2.boundingRect(contours[i])
            print(w)
            #cv2.rectangle(img_output, (x,y), (x+w, y+h), (0, 255, 0), 1)
            output = cv2.resize(img_output[y:y+h, x:x+w], (28,28), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite("./data/" + str(label) + "_" + str(count) + ".jpg", output)
            count += 1
    #cv2.drawContours(img, contours, -1, (0, 255, 0), 2)
    #cv2.imshow('test', img_output)
    #cv2.waitKey(0)
def numbers_to_number():
    for i in range(images_numbers):
        split_numbers(i, "./numbers/" + str(i) + ".jpg")
if __name__ == '__main__':
    #catch confirm image from website
    catch_data()
    #split confirm image into six single number
    numbers_to_number()
