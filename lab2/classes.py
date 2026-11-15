import dataclasses
import math

@dataclasses.dataclass
class Point: #
    x : int = 0
    y : int = 0

class Rectangle:
    def __init__(self, upper_left: Point, lower_right: Point):
        self._upper_left = upper_left
        self._lower_right = lower_right
    @property
    def width(self):
        return self._lower_right.x - self._upper_left.x
    @property
    def height(self):
        return self._lower_right.y - self._upper_left.y
    @property
    def area(self):
        return self.width * self.height
    @property
    def upper_left(self) -> Point:
        return self._upper_left
    @property
    def lower_right(self) -> Point:
        return self._lower_right
    def __repr__(self):
        return "(%d, %d); (%d, %d)" % (
            self.upper_left.x, self.upper_left.y,
            self.lower_right.x, self.lower_right.y
        )

def find_bounding_box(labels, number_labels):
    bounding_boxes = [Rectangle(Point(math.inf, math.inf), Point(0,0))
                      for i in range(number_labels)]
    for i in range(labels.shape[0]): # строки (y)
        for j in range(labels.shape[1]): # столбцы (x)
            lbl = labels[i, j] # номер метки
            # (и прямоугольника, который
            # будет соответствовать этой метке)
            top_left = bounding_boxes[lbl].upper_left
            bot_right = bounding_boxes[lbl].lower_right

            # для верхнего левого угла
            if top_left.y > i: # ищется минимальное значение строки
                top_left.y = i
            if top_left.x > j: # и столбца
                top_left.x = j
            # для нижнего правого угла
            if bot_right.y < i: # ищется максимальное значение строки
                bot_right.y = i
            if bot_right.x < j: # и столбца
                bot_right.x = j
    return bounding_boxes