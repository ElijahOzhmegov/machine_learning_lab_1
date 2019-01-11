# Импортируем все необходимые библиотеки. sklearn устанавливается через pip или conda
import numpy as np
import matplotlib.pylab as pl
from sklearn import neighbors, datasets
from sklearn.model_selection import train_test_split
import calculate_accuracy as ca

# definitely, I should learn sklearn, so
# it has an entire bunch of extremely useful tools

# Импортируем данные. Спасибо sklearn, в нём есть модуль datasets, содержащий известные базы данных.
iris = datasets.load_iris()
# Берем все входные примеры из базы, но для каждого примера берем только первые два признака (т.е. берем подмассив)
X = iris.data[:,:2]
# Т.к. взяли все входные примеры, берем все выходные значения - класс ириса.
Y = iris.target

new_X, new_Y = ca.get_rid_of_duplicates(X, Y)

# шаг сетки
h = .02


# a random split into training and test sets
X_train_a, X_test_a, Y_train_a, Y_test_a = train_test_split(X, Y, test_size=0.3, random_state=15)

X = new_X
Y = new_Y

X_train_b, X_test_b, Y_train_b, Y_test_b = train_test_split(X, Y, test_size=0.3, random_state=15)

file = open("results/result_table.tex", 'w')

for k in range(3, 30, 2):
    # И снова спасибо sklearn: создаем экземпляр классификатора kNN
    # Мы берем все параметры по умолчанию, а вообще их можно задать
    knn1 = neighbors.KNeighborsClassifier(n_neighbors=k)
    knn2 = neighbors.KNeighborsClassifier(n_neighbors=k, weights='distance')
    knn3 = neighbors.KNeighborsClassifier(n_neighbors=k)
    knn4 = neighbors.KNeighborsClassifier(n_neighbors=k, weights='distance')

    # "Обучаем" классификатор на наших данных
    knn1.fit(X_train_a, Y_train_a)
    knn2.fit(X_train_a, Y_train_a)
    knn3.fit(X_train_b, Y_train_b)
    knn4.fit(X_train_b, Y_train_b)

    # Рассчитаем точность классификации (раскомментируйте три следующие строки и впишите код для расчета точности)
    acc1 = ca.get_accuracy(knn1, X_test_a, Y_test_a)
    acc2 = ca.get_accuracy(knn2, X_test_a, Y_test_a)
    acc3 = ca.get_accuracy(knn3, X_test_b, Y_test_b)
    acc4 = ca.get_accuracy(knn4, X_test_b, Y_test_b)

    # result = "Точность: {} %".format(np.round(acc1*100, 3))
    # print(result)

    file.write(str(k) + ' & ' + str(np.round(acc1 * 100, 3)) +
                        ' & ' + str(np.round(acc2 * 100, 3)) +
                        ' & ' + str(np.round(acc3 * 100, 3)) +
                        ' & ' + str(np.round(acc4 * 100, 3)) + " \\\\" + ' ' + '\n')

file.close()

# Построим диаграмму, иллюстрирующую полученные классы.
# Для этого присвоим каждой ячейке сетки [x1_min, x1_max]x[x2_min, x2_max]
# цвет классa, которому она принадлежит.

# Находим x1_min, x1_max, x2_min, x2_max - это минимальные и максимальные значения
# двух признаков, содержащихся в X. Слагаемое 0.5 - для отступа на диаграмме.
x1_min, x1_max = X[:,0].min() - .5, X[:,0].max() + .5
x2_min, x2_max = X[:,1].min() - .5, X[:,1].max() + .5
# Строим сетки, т.е. матрицы, содержащие значения признаков в диапазоне от min до max
x1, x2 = np.meshgrid(np.arange(x1_min, x1_max, h), np.arange(x2_min, x2_max, h))
# Скармливаем эти матрицы классификатору, на выходе получаем значение класca для каждой ячейки
Z = knn4.predict(np.c_[x1.ravel(), x2.ravel()])

# Приведем форму результата к матрице такого же размера, что и матрицы признаков
Z = Z.reshape(x1.shape)

# Строим цветную диаграмму зависимости класса (Z) от признаков (x1, x2).
# Класc обозначен цветом.
pl.figure(2, figsize=(12, 9))
pl.set_cmap(pl.cm.Paired)
pl.pcolormesh(x1, x2, Z)

# Также построим точки, показывающие обучающие примеры.
# Это будет диаграмма рассеяния (scater plot)
pl.scatter(X[:,0], X[:,1],c=Y, edgecolors="b")
# Подпишем оси
pl.xlabel('Sepal length')
pl.ylabel('Sepal width')
# Укажем границы диаграммы
pl.xlim(x1.min(), x1.max())
pl.ylim(x2.min(), x2.max())
pl.xticks(())
pl.yticks(())
# Выведем, что получилось
pl.show()
