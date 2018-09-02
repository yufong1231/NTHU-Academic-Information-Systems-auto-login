import tensorflow as tf
import cv2
import numpy as np
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.alert import Alert
from selenium.webdriver.support import expected_conditions as EC


def predict_single_number(image):

    with tf.Session() as sess:
        save_path = './model/test_model-900.meta'

        saver = tf.train.import_meta_graph(save_path)

        saver.restore(sess, "./model/test_model-900")

        x = tf.get_collection("input")[0]
        y = tf.get_collection("output")[0]

        result = sess.run(y, feed_dict = {x: image.reshape((-1, 28*28))})

        answer = '0'
        if result[0] == 0:
            answer = '9'
        elif result[0] == 1:
            answer = '0'
        elif result[0] == 2:
            answer = '7'
        elif result[0] == 3:
            answer = '6'
        elif result[0] == 4:
            answer = '1'
        elif result[0] == 5:
            answer = '8'
        elif result[0] == 6:
            answer = '4'
        elif result[0] == 7:
            answer = '3'
        elif result[0] == 8:
            answer = '2'
        else:
            answer = '5'

        return answer

def split_image(image):
    image = np.array(image, dtype=np.uint8)
    single_numbers = np.zeros([6, 28, 28])
    number_pirority = np.zeros([6])
    correct_detect = False
    # origin image -> gray -> threshold
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, img_threshold = cv2.threshold(img_gray, 150, 255, cv2.THRESH_BINARY_INV);
    ret, img_output = cv2.threshold(img_gray, 150, 255, cv2.THRESH_BINARY);
    #find contours
    image_contours, contours, hierarchy = cv2.findContours(img_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #error case
    if len(contours) > 6:
        print("Error : detect too many coutours!")
        correct_detect = False
    #two numbers connect case
    elif len(contours) < 6 and len(contours) > 0:
        correct_detect = True
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
                #cv2.imwrite("./data/" + str(label) + "_" + str(count) + ".jpg", output)
                single_numbers[count] = output
                number_pirority[count] = x
                count += 1
                output = cv2.resize(img_output[y:y+h, int(x+w/2):x+w], (28,28), interpolation=cv2.INTER_CUBIC)
                #cv2.imwrite("./data/" + str(label) + "_" + str(count) + ".jpg", output)
                single_numbers[count] = output
                number_pirority[count] = int(x+w/2)
                count += 1
            else:
                #cv2.rectangle(img_output, (x,y), (x+w, y+h), (0, 255, 0), 1)
                output = cv2.resize(img_output[y:y+h, x:x+w], (28,28), interpolation=cv2.INTER_CUBIC)
                #cv2.imwrite("./data/" + str(label) + "_" + str(count) + ".jpg", output)
                single_numbers[count] = output
                number_pirority[count] = x
                count += 1
    #normal case (can detect accurate six numbers)
    else:
        count = 0
        correct_detect = True
        for i in range(0,len(contours)):
            x, y, w, h = cv2.boundingRect(contours[i])
            #cv2.rectangle(img_output, (x,y), (x+w, y+h), (0, 255, 0), 1)
            output = cv2.resize(img_output[y:y+h, x:x+w], (28,28), interpolation=cv2.INTER_CUBIC)
            #cv2.imwrite("./data/" + str(label) + "_" + str(count) + ".jpg", output)
            single_numbers[count] = output
            number_pirority[count] = x
            count += 1
    #cv2.drawContours(img, contours, -1, (0, 255, 0), 2)
    #cv2.imshow('test', img_output)
    #cv2.waitKey(0)
    sorted_number = sorted(number_pirority)
    sort_single_numbers = np.zeros([6, 28, 28])
    for i in range(6):
        for j in range(6):
            if sorted_number[i] == number_pirority[j]:
                sort_single_numbers[i] = single_numbers[j]

    return correct_detect, sort_single_numbers


if __name__ == '__main__':

    user_account = input('Enter Your Account : ')
    user_password = input('Enter Your Password : ')

    driver = webdriver.Chrome()
    driver.set_window_size(1200, 900)
    driver.get('https://www.ccxp.nthu.edu.tw/ccxp/INQUIRE/')
    #detect user account or password correct
    account_error = False

    while(1):
        #get confrim image
        driver.save_screenshot('./screenshot.jpg')
        screenshot = cv2.imread('./screenshot.jpg')
        img = screenshot[514:576, 85:285]
        #split confrim image to six single numbers
        correct_detect, single_numbers = split_image(img)

        if correct_detect == False:
            print('cannot detect correct comfrim password, refresh the window and try again')
            driver.refresh()
        else:
            num = [0, 0 ,0 ,0 ,0 ,0]
            for i in range(6):
                num[i] = predict_single_number(single_numbers[i])

            str = ''.join(num)
            print(str)
            #fill in inputs
            account = driver.find_element_by_name('account')
            account.clear()
            account.send_keys(user_account)

            password = driver.find_element_by_name('passwd')
            password.clear()
            password.send_keys(user_password)

            comfrim_numbers = driver.find_element_by_name('passwd2')
            comfrim_numbers.clear()
            comfrim_numbers.send_keys(str)

            submit = driver.find_element_by_name('Submit')
            submit.click()

            time.sleep(1)
            #detect whether login is correct
            result = EC.alert_is_present()(driver)
            if result:
                if result.text == '帳號或密碼錯誤, 請重新輸入!':
                    account_error = True
                    break
                print('comfrim password error, refresh the window and try again')
                alert = Alert(driver)
                alert.accept()
            else:
                print('correct login')
                break
            #alert.accept();
    if account_error == True:
        print('----------------------------------------------')
        print('Your account or password incorrect, please check')
        print('----------------------------------------------')
    else:
        print('----------------------------------------------')
        print('welcome to NTHU Academic Information Systems!')
        print('----------------------------------------------')
    #time.sleep(1)
    #driver.close()
