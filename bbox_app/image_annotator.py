import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QAction, QFileDialog, QInputDialog
from PyQt5.QtGui import QPixmap, QPainter, QPen, QGuiApplication, QKeySequence, QCursor, QBrush, QColor
from PyQt5.QtCore import Qt, QPoint, QRect, pyqtSignal

# Define constants for handles and their properties
(
    H_TOP_LEFT, H_TOP_CENTER, H_TOP_RIGHT,
    H_MIDDLE_LEFT, H_MIDDLE_RIGHT,
    H_BOTTOM_LEFT, H_BOTTOM_CENTER, H_BOTTOM_RIGHT
) = range(8)

HANDLE_SIZE = 8

# Map handle indices to cursor shapes
HANDLE_CURSORS = {
    H_TOP_LEFT: Qt.SizeFDiagCursor,
    H_TOP_CENTER: Qt.SizeVerCursor,
    H_TOP_RIGHT: Qt.SizeBDiagCursor,
    H_MIDDLE_LEFT: Qt.SizeHorCursor,
    H_MIDDLE_RIGHT: Qt.SizeHorCursor,
    H_BOTTOM_LEFT: Qt.SizeBDiagCursor,
    H_BOTTOM_CENTER: Qt.SizeVerCursor,
    H_BOTTOM_RIGHT: Qt.SizeFDiagCursor,
}

class ImageLabel(QLabel):
    """
    自訂 QLabel 以處理滑鼠事件來繪製矩形。
    """
    # 信號：當滑鼠在圖片上移動時發送原始座標字串
    mouse_position_changed = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.begin = QPoint()
        self.end = QPoint()
        self.drawing = False
        self.rect_list = []
        self.selected_rect_index = None
        self.moving = False
        self.resizing = False
        self.resize_handle_index = None
        self.drag_start_position = QPoint()
        self.setMouseTracking(True) # 持續追蹤滑鼠位置
        self.setFocusPolicy(Qt.StrongFocus) # 設定焦點策略以接收鍵盤事件
        self.original_pixmap_size = None
        self.scaled_pixmap_size = None

    def mousePressEvent(self, event):
        """滑鼠按下事件：處理選擇、移動、調整大小或開始繪製。"""
        if event.button() == Qt.LeftButton:
            # 優先檢查是否點擊在控制點上
            handle_index = self._get_handle_at(event.pos())
            if handle_index is not None:
                self.resizing = True
                self.resize_handle_index = handle_index
                self.moving = False
                self.drawing = False
                self.update()
                return

            # 檢查是否點擊在現有矩形上 (從最上層的框開始檢查)
            for i in range(len(self.rect_list) - 1, -1, -1):
                if self.rect_list[i].contains(event.pos()):
                    self.selected_rect_index = i
                    self.drawing = False
                    self.resizing = False
                    self.moving = True
                    self.drag_start_position = event.pos()
                    self.update()
                    return # 找到並選中，結束事件處理

            # 如果沒有點擊在任何矩形上，則取消選擇並開始繪製新矩形
            self.selected_rect_index = None
            self.moving = False
            self.resizing = False
            self.begin = event.pos()
            self.end = self.begin
            self.drawing = True
            self.update() # 重繪以清除舊的選擇高亮

    def mouseMoveEvent(self, event):
        """滑鼠移動事件：處理拖曳、繪製、調整大小並發送座標信號。"""
        pos = event.pos()

        # --- 更新滑鼠游標形狀 ---
        # 只有在沒有進行任何操作時才更新游標
        if not self.drawing and not self.moving and not self.resizing:
            handle_index = self._get_handle_at(pos)
            if handle_index is not None:
                cursor = QCursor(HANDLE_CURSORS[handle_index])
                self.setCursor(cursor)
            elif self.selected_rect_index is not None and self.rect_list[self.selected_rect_index].contains(pos):
                self.setCursor(QCursor(Qt.SizeAllCursor))
            else:
                self.unsetCursor()

        # --- 座標顯示邏輯 ---
        if self.original_pixmap_size and self.scaled_pixmap_size and self.scaled_pixmap_size.width() > 0 and self.scaled_pixmap_size.height() > 0:
            scale_x = self.original_pixmap_size.width() / self.scaled_pixmap_size.width()
            scale_y = self.original_pixmap_size.height() / self.scaled_pixmap_size.height()

            orig_x = int(pos.x() * scale_x)
            orig_y = int(pos.y() * scale_y)

            # 確保座標在原始圖片範圍內
            if 0 <= orig_x < self.original_pixmap_size.width() and 0 <= orig_y < self.original_pixmap_size.height():
                coord_text = f"原始座標: ({orig_x}, {orig_y})"
                self.mouse_position_changed.emit(coord_text)
            else:
                # 滑鼠在標籤上，但在計算出的圖片區域之外
                self.mouse_position_changed.emit("")

        # --- 拖曳、繪製或調整大小邏輯 ---
        if self.resizing and self.selected_rect_index is not None:
            # 調整選中框的大小
            rect = self.rect_list[self.selected_rect_index]
            if self.resize_handle_index == H_TOP_LEFT:
                rect.setTopLeft(pos)
            elif self.resize_handle_index == H_TOP_CENTER:
                rect.setTop(pos.y())
            elif self.resize_handle_index == H_TOP_RIGHT:
                rect.setTopRight(pos)
            elif self.resize_handle_index == H_MIDDLE_LEFT:
                rect.setLeft(pos.x())
            elif self.resize_handle_index == H_MIDDLE_RIGHT:
                rect.setRight(pos.x())
            elif self.resize_handle_index == H_BOTTOM_LEFT:
                rect.setBottomLeft(pos)
            elif self.resize_handle_index == H_BOTTOM_CENTER:
                rect.setBottom(pos.y())
            elif self.resize_handle_index == H_BOTTOM_RIGHT:
                rect.setBottomRight(pos)
            
            # 確保矩形有效 (左上角 < 右下角)，並更新列表
            self.rect_list[self.selected_rect_index] = rect.normalized()
            self.update()
        elif self.moving and self.selected_rect_index is not None:
            # 拖曳移動選中的框
            offset = pos - self.drag_start_position
            self.rect_list[self.selected_rect_index].translate(offset)
            self.drag_start_position = pos
            self.update()
        elif self.drawing:
            # 繪製新框
            self.end = pos
            self.update() # 觸發 paintEvent

    def mouseReleaseEvent(self, event):
        """滑鼠放開事件：完成繪製、移動或調整大小。"""
        if event.button() == Qt.LeftButton:
            if self.resizing:
                self.resizing = False
                self.unsetCursor()
                self.update()
            elif self.moving:
                self.moving = False
                self.unsetCursor()
                self.update()
            elif self.drawing:
                self.drawing = False
                rect = QRect(self.begin, self.end).normalized()
                if rect.width() > 4 and rect.height() > 4:
                    self.rect_list.append(rect)
                    self.selected_rect_index = len(self.rect_list) - 1 # 新增後自動選中
                self.update()

    def undo(self):
        """移除最後一個繪製的矩形。"""
        if self.rect_list:
            self.rect_list.pop()
            self.selected_rect_index = None # 清除選擇
            self.update()

    def keyPressEvent(self, event):
        """鍵盤按下事件：處理刪除鍵。"""
        # 如果按下 Delete 鍵且有選中的矩形
        if event.key() == Qt.Key_Delete and self.selected_rect_index is not None:
            self.rect_list.pop(self.selected_rect_index)
            self.selected_rect_index = None # 清除選擇
            self.update()
        super().keyPressEvent(event)

    def leaveEvent(self, event):
        """滑鼠離開事件：發送空字串以清除座標顯示。"""
        self.mouse_position_changed.emit("")
        super().leaveEvent(event)

    def _get_handles(self, rect):
        """計算給定矩形的八個控制點的位置。"""
        if rect.isNull():
            return []
        
        hs = HANDLE_SIZE // 2
        
        handles = [
            # Top-left, top-center, top-right
            QRect(rect.left() - hs, rect.top() - hs, HANDLE_SIZE, HANDLE_SIZE),
            QRect(rect.center().x() - hs, rect.top() - hs, HANDLE_SIZE, HANDLE_SIZE),
            QRect(rect.right() - hs, rect.top() - hs, HANDLE_SIZE, HANDLE_SIZE),
            # Middle-left, middle-right
            QRect(rect.left() - hs, rect.center().y() - hs, HANDLE_SIZE, HANDLE_SIZE),
            QRect(rect.right() - hs, rect.center().y() - hs, HANDLE_SIZE, HANDLE_SIZE),
            # Bottom-left, bottom-center, bottom-right
            QRect(rect.left() - hs, rect.bottom() - hs, HANDLE_SIZE, HANDLE_SIZE),
            QRect(rect.center().x() - hs, rect.bottom() - hs, HANDLE_SIZE, HANDLE_SIZE),
            QRect(rect.right() - hs, rect.bottom() - hs, HANDLE_SIZE, HANDLE_SIZE)
        ]
        return handles

    def _get_handle_at(self, pos):
        """檢查給定位置是否有控制點，返回其索引。"""
        if self.selected_rect_index is not None:
            selected_rect = self.rect_list[self.selected_rect_index]
            handles = self._get_handles(selected_rect)
            for i, handle in enumerate(handles):
                if handle.contains(pos):
                    return i
        return None

    def paintEvent(self, event):
        """繪製事件：在圖片上繪製所有矩形。"""
        super().paintEvent(event)
        painter = QPainter(self)

        # --- 座標轉換比例計算 ---
        scale_x, scale_y = 1.0, 1.0
        can_calculate_scale = (self.original_pixmap_size and 
                               self.scaled_pixmap_size and 
                               self.scaled_pixmap_size.width() > 0 and 
                               self.scaled_pixmap_size.height() > 0)
        if can_calculate_scale:
            scale_x = self.original_pixmap_size.width() / self.scaled_pixmap_size.width()
            scale_y = self.original_pixmap_size.height() / self.scaled_pixmap_size.height()

        # --- 繪製所有已儲存的矩形及座標文字 ---
        pen_saved = QPen(Qt.red, 2, Qt.SolidLine)
        pen_selected = QPen(Qt.yellow, 3, Qt.SolidLine)
        mask_brush = QBrush(QColor(255, 255, 0, 70)) # 半透明黃色遮罩
        handle_brush = QBrush(Qt.yellow)

        for i, rect in enumerate(self.rect_list):
            # 根據是否被選中來決定繪製樣式
            if i == self.selected_rect_index:
                # 繪製選中框: 遮罩 -> 邊框 -> 控制點
                painter.setBrush(mask_brush)
                painter.setPen(Qt.NoPen)
                painter.drawRect(rect)

                painter.setBrush(Qt.NoBrush)
                painter.setPen(pen_selected)
                painter.drawRect(rect)

                painter.setBrush(handle_brush)
                painter.setPen(Qt.NoPen)
                handles = self._get_handles(rect)
                for handle in handles:
                    painter.drawRect(handle)
            else:
                # 繪製未選中框
                painter.setBrush(Qt.NoBrush)
                painter.setPen(pen_saved)
                painter.drawRect(rect)

            # 為所有框繪製座標文字
            painter.setPen(pen_saved) # 座標文字統一使用紅色
            if can_calculate_scale:
                # 將顯示座標轉換回原始座標以供顯示
                orig_x1 = int(rect.left() * scale_x)
                orig_y1 = int(rect.top() * scale_y)
                orig_x2 = int(rect.right() * scale_x)
                orig_y2 = int(rect.bottom() * scale_y)

                coord_text = f"({orig_x1}, {orig_y1}, {orig_x2}, {orig_y2})"
                text_pos = rect.topLeft() - QPoint(0, 5) # 將文字放在框的左上角上方

                # 如果文字會超出頂部邊界，則將其移到框內
                if text_pos.y() < 10:
                    text_pos.setY(rect.top() + 15)

                painter.drawText(text_pos, coord_text)

        # --- 繪製當前正在畫的矩形 (使用不同顏色和虛線以作區分) ---
        if self.drawing:
            pen_drawing = QPen(Qt.blue, 2, Qt.DashLine)
            painter.setPen(pen_drawing)
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(QRect(self.begin, self.end))

    def clear_rects(self):
        """清除所有矩形。"""
        self.rect_list.clear()
        self.selected_rect_index = None
        self.update()

    def add_rect_from_original_coords(self, orig_x1, orig_y1, orig_x2, orig_y2):
        """從原始圖片座標新增一個矩形。"""
        if not self.original_pixmap_size or not self.scaled_pixmap_size or self.scaled_pixmap_size.width() == 0 or self.scaled_pixmap_size.height() == 0:
            print("無法進行座標轉換，圖片尺寸資訊不完整。")
            return

        # 計算從原始座標到顯示座標的縮放比例
        scale_x = self.scaled_pixmap_size.width() / self.original_pixmap_size.width()
        scale_y = self.scaled_pixmap_size.height() / self.original_pixmap_size.height()

        # 將原始座標轉換為顯示座標
        scaled_x1 = int(orig_x1 * scale_x)
        scaled_y1 = int(orig_y1 * scale_y)
        scaled_x2 = int(orig_x2 * scale_x)
        scaled_y2 = int(orig_y2 * scale_y)

        # 建立 QRect 並加入列表
        new_rect = QRect(QPoint(scaled_x1, scaled_y1), QPoint(scaled_x2, scaled_y2)).normalized()
        
        if new_rect.width() > 0 and new_rect.height() > 0:
            self.rect_list.append(new_rect)
            self.selected_rect_index = len(self.rect_list) - 1 # 選中新增的框
            self.update()
            print(f"已從座標新增標註框: ({orig_x1}, {orig_y1}, {orig_x2}, {orig_y2})")
        else:
            print("從座標建立的矩形無效 (寬或高為0)。")

class MainWindow(QMainWindow):
    """
    主視窗應用程式。
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("圖片標註工具")
        self.image_path = ""
        self.status_message = "請透過 '檔案' -> '開啟圖片' 來載入一張圖片。"

        # 建立用於顯示和繪圖的 ImageLabel
        self.image_label = ImageLabel(self)
        self.setCentralWidget(self.image_label)

        # 建立狀態列並連接信號
        self.statusBar().showMessage(self.status_message)
        self.image_label.mouse_position_changed.connect(self.update_status_bar)

        self._create_menus()

        # 初始視窗大小
        self.resize(800, 600)
        self._center_window()

    def update_status_bar(self, text):
        """更新狀態列的訊息。如果文字為空，則顯示預設訊息。"""
        if text:
            self.statusBar().showMessage(text)
        else:
            self.statusBar().showMessage(self.status_message)

    def _create_menus(self):
        """建立功能表列。"""
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("檔案")

        # 開啟圖片動作
        open_action = QAction("開啟圖片", self)
        open_action.triggered.connect(self.open_image)
        file_menu.addAction(open_action)

        # 儲存標註動作
        save_action = QAction("儲存標註", self)
        save_action.triggered.connect(self.save_annotations)
        file_menu.addAction(save_action)

        # 建立編輯選單和復原動作
        edit_menu = menu_bar.addMenu("編輯")
        undo_action = QAction("復原", self)
        undo_action.setShortcut(QKeySequence.Undo) # 標準的 Ctrl+Z 快捷鍵
        undo_action.triggered.connect(self.image_label.undo)
        edit_menu.addAction(undo_action)

        # 從座標新增標註框動作
        add_bbox_action = QAction("從座標新增標註框", self)
        add_bbox_action.setShortcut("Ctrl+N")
        add_bbox_action.triggered.connect(self.add_bbox_from_coords)
        edit_menu.addAction(add_bbox_action)

    def _center_window(self):
        """將視窗置於螢幕中央。"""
        screen = QGuiApplication.primaryScreen().geometry()
        x = (screen.width() - self.width()) // 2
        y = (screen.height() - self.height()) // 2
        self.move(x, y)

    def add_bbox_from_coords(self):
        """開啟對話框讓使用者輸入座標以新增標註框。"""
        # 檢查是否已載入圖片
        if not self.image_label.original_pixmap_size or not self.image_label.scaled_pixmap_size:
            self.statusBar().showMessage("請先載入圖片再新增標註框。")
            return

        # 顯示輸入對話框
        text, ok = QInputDialog.getText(self, "從座標新增標註框", "請輸入原始座標 (x1, y1, x2, y2)，以逗號分隔:")
        
        if ok and text:
            try:
                # 解析並驗證輸入
                coords = [int(c.strip()) for c in text.split(',')]
                if len(coords) != 4:
                    raise ValueError("需要四個座標值。")
                
                x1, y1, x2, y2 = coords
                
                # 呼叫 ImageLabel 的方法來新增矩形
                self.image_label.add_rect_from_original_coords(x1, y1, x2, y2)
                self.status_message = f"已從座標新增標註框: ({x1}, {y1}, {x2}, {y2})"
                self.statusBar().showMessage(self.status_message)
            except ValueError as e:
                self.statusBar().showMessage(f"輸入格式錯誤，請使用 'x1,y1,x2,y2' 格式。 ({e})")

    def open_image(self):
        """開啟並顯示圖片檔案，並根據螢幕大小調整圖片尺寸。"""
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "選擇圖片檔案", "",
                                                  "圖片檔案 (*.png *.jpg *.jpeg *.bmp)", options=options)
        if file_path:
            self.image_path = file_path
            pixmap = QPixmap(self.image_path)
            if pixmap.isNull():
                print("無法載入圖片")
                return

            # 清除舊的標註
            self.image_label.clear_rects()

            # 根據螢幕可用大小縮放圖片以適應視窗
            screen_geometry = QGuiApplication.primaryScreen().availableGeometry()
            original_size = pixmap.size()

            # 為了避免視窗太大，可以設定一個邊界
            margin = 100
            screen_geometry.setWidth(screen_geometry.width() - margin)
            screen_geometry.setHeight(screen_geometry.height() - margin)

            scaled_pixmap = pixmap.scaled(screen_geometry.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)

            # 儲存原始和縮放後的大小以供座標轉換
            self.image_label.original_pixmap_size = original_size
            self.image_label.scaled_pixmap_size = scaled_pixmap.size()

            self.image_label.setPixmap(scaled_pixmap)
            self.resize(scaled_pixmap.size()) # 調整視窗大小以符合縮放後的圖片
            self._center_window()

            self.status_message = f"已載入: {self.image_path}"
            self.statusBar().showMessage(self.status_message)

    def save_annotations(self):
        """將轉換回原始圖片尺寸的矩形座標儲存到 txt 檔案。"""
        if not self.image_label.rect_list:
            print("沒有任何標註可以儲存。")
            return

        if not self.image_label.original_pixmap_size or not self.image_label.scaled_pixmap_size:
            print("沒有載入圖片，無法儲存標註。")
            return

        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(self, "儲存標註檔案", "",
                                                   "文字檔案 (*.txt)", options=options)

        if file_path:
            original_size = self.image_label.original_pixmap_size
            scaled_size = self.image_label.scaled_pixmap_size

            if scaled_size.width() == 0 or scaled_size.height() == 0:
                print("錯誤：縮放後的圖片尺寸為0。")
                return

            scale_x = original_size.width() / scaled_size.width()
            scale_y = original_size.height() / scaled_size.height()

            try:
                with open(file_path, 'w') as f:
                    for rect in self.image_label.rect_list:
                        # 將顯示座標轉換回原始座標
                        x1 = int(rect.left() * scale_x)
                        y1 = int(rect.top() * scale_y)
                        x2 = int(rect.right() * scale_x)
                        y2 = int(rect.bottom() * scale_y)
                        f.write(f"{x1},{y1},{x2},{y2}\n")
                print(f"標註已成功儲存至: {file_path}")
            except Exception as e:
                print(f"儲存檔案時發生錯誤: {e}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_win = MainWindow()
    main_win.show()
    sys.exit(app.exec_())
