from sklearn.datasets import fetch_lfw_people
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score,confusion_matrix
import seaborn as sb

faces = fetch_lfw_people(min_faces_per_person=60)

# print(faces.target_names)
# print(faces.images.shape)

# fig, ax = plt.subplots(3, 5)

# for i, axi in enumerate(ax.flat):
#     axi.imshow(faces.images[i], cmap='bone')
#     axi.set(xticks=[], yticks=[])
#     axi.set_ylabel(faces.target_names[faces.target[i]].split()[-1],
#                    color='black')

# plt.show()

#reduce and crete model
pca = PCA(n_components=150,svd_solver="randomized",whiten=True)
svc = SVC(kernel="rbf",class_weight="balanced")

model = make_pipeline(pca,svc)

x_train,x_test,y_train,y_test = train_test_split(faces.data,faces.target,random_state=40)

param = {
    "svc__C": [1, 5, 10, 50],
    "svc__gamma": [0.0001, 0.0005, 0.001, 0.005]
}


grid = GridSearchCV(model,param)
grid.fit(x_train,y_train)
# print("ค่า C และ gamma ที่ดีที่สุด : ",grid.best_params_)
# print("ค่า C และ gamma เวลาเอาไปใช้ : ",grid.best_estimator_)

model = grid.best_estimator_

# predict
y_pred = model.predict(x_test)

# show real image & predict name
# fig, ax = plt.subplots(4, 6)

# for i, axi in enumerate(ax.flat):
#     axi.imshow(x_test[i].reshape(62, 47), cmap='bone')
#     axi.set(xticks=[], yticks=[])
#     axi.set_ylabel(
#         faces.target_names[y_pred[i]].split()[-1],
#         color='green' if y_pred[i] == y_test[i] else 'red'
#     )

# plt.show()
# print("Accuracy = ", accuracy_score(y_test, y_pred) * 100)

mat = confusion_matrix(y_test, y_pred)

sb.heatmap(
    mat.T,
    square=True,
    annot=True,
    fmt='d',
    cbar=False,
    xticklabels=faces.target_names,
    yticklabels=faces.target_names
)

plt.xlabel("True Data")
plt.ylabel("Predict Data")
plt.show()
