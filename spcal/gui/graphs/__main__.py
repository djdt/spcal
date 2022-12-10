from PySide6 import QtGui, QtWidgets

from spcal.gui.graphs import color_schemes


def main() -> int:  # test colors
    app = QtWidgets.QApplication()
    scene = QtWidgets.QGraphicsScene(
        -50,
        -50,
        200 + 100 * max(len(v) for v in color_schemes.values()),
        100 + 100 * len(color_schemes),
    )
    view = QtWidgets.QGraphicsView(scene)

    yy = 0
    for name, colors in color_schemes.items():
        label = QtWidgets.QGraphicsTextItem(name)
        label.setPos(0, yy)
        view.scene().addItem(label)
        xx = 0
        for color in colors:
            xx += 100
            rect = QtWidgets.QGraphicsRectItem(xx, yy, 50, 50)
            rect.setBrush(QtGui.QBrush(color))
            view.scene().addItem(rect)
        yy += 100

    view.show()
    return app.exec()


if __name__ == "__main__":
    main()
