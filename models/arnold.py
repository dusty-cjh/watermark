import numpy as np


class ArnoldTransform:
    def __init__(self, a=1, b=1):
        self.A = np.matrix([
            [1, a],
            [b, a*b+1],
        ])
        self.A_ = self.A.I

    def _trans(self, origin: np.ndarray, kernel=None, count=3):
        if count > 0:
            origin = self._trans(origin, kernel=kernel, count=count-1)
        else:
            return origin

        dest = origin.copy()
        rows, cols = origin.shape[:2]
        assert rows == cols, 'rows != cols'
        A = kernel
        r, c = np.meshgrid(np.arange(rows), np.arange(cols))
        grid = np.array([r, c]).transpose(1,2,0)
        grid = grid.reshape((-1, 2)).T
        new_grid = np.asarray(np.mod(A * grid, rows), dtype=np.int64)
        dest[new_grid[0, :], new_grid[1, :]] = origin[grid[0, :], grid[1, :]]
        return dest

    def transform(self, origin: np.ndarray, count=3):
        # 1. get the minimum edge of the rectangle
        # 2. split rectangle to squares from top-left to bottom-right
        # 3. do transformation for these squares orderly
        rows, cols = origin.shape[:2]
        if rows > cols:
            return self.transform(origin.swapaxes(0, 1), count=count).swapaxes(0, 1)
        elif rows == cols:
            return self._trans(origin, kernel=self.A, count=count)

        # todo: 用动态规划来解决，不过动态规划可能会引入鲁棒性的降低

        # split squares
        square_count = int(np.ceil(cols / rows))
        assert square_count >= 2, 'square_count < 2'
        for _ in range(count):
            for i in range(square_count - 1):
                square = origin[:, i*rows:(i+1)*rows]
                dest_square = self.transform(square, count=1)
                origin[:, i*rows:(i+1)*rows] = dest_square
            square = origin[:, -rows:]
            origin[:, -rows:] = self.transform(square, count=1)
        return origin

    def inverse_transform(self, origin: np.ndarray, count=3):
        rows, cols = origin.shape[:2]
        if rows > cols:
            return self.inverse_transform(origin.swapaxes(0, 1), count=count).swapaxes(0, 1)
        elif rows == cols:
            return self._trans(origin, kernel=self.A_, count=count)

        # todo: 用动态规划来解决，不过动态规划可能会引入鲁棒性的降低

        # split squares
        square_count = int(np.ceil(cols / rows))
        assert square_count >= 2, 'square_count < 2'
        for _ in range(count):
            square = origin[:, -rows:]
            origin[:, -rows:] = self.inverse_transform(square, count=1)
            for i in np.arange(square_count - 2, -1, step=-1):
                square = origin[:, i*rows:(i+1)*rows]
                origin[:, i*rows:(i+1)*rows] = self.inverse_transform(square, count=1)
        return origin

        # return self._trans(origin, kernel=self.A_, count=count)
