# import cv2

# img_path = "image/yoga.jpg"
# img = cv2.imread(img_path)

# if img is None:
#     print("Khong the doc hinh anh tu duong dan:",img_path)
# else:
#     cv2.imshow("image", img)
#     while True:
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             cv2.destroyAllWindows()


from sklearn.preprocessing import LabelEncoder

# Mã hóa nhãn
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Lấy danh sách nhãn và thứ tự tương ứng
label_list = label_encoder.classes_
