import os
import cv2
import numpy as np


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def read_image(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {path}")
    return img


def resize_to_target(img, target_shape):
    target_h, target_w = target_shape[:2]
    return cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)


def normalize_to_uint8(x):
    x = x.astype(np.float32)
    min_val = np.min(x)
    max_val = np.max(x)

    if max_val - min_val < 1e-8:
        return np.zeros_like(x, dtype=np.uint8)

    x = (x - min_val) / (max_val - min_val)
    x = (x * 255).astype(np.uint8)
    return x


def heatmap_to_response(heatmap):
    """
    将已经着色的 heatmap 近似转成响应强度图。
    注意：如果你有原始 attention 数值，用原始 attention 会更准确。
    """
    gray = cv2.cvtColor(heatmap, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    gray = normalize_to_uint8(gray)
    return gray


def get_top_response_mask(response, top_percent=20):
    """
    取响应最高的 top_percent 区域作为显著区域
    """
    threshold = np.percentile(response, 100 - top_percent)
    mask = (response >= threshold).astype(np.uint8) * 255

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask


def draw_contours_on_image(image, mask1, mask2):
    """
    在原图上画两个 heatmap 的高响应轮廓
    红色：heatmap1
    青色：heatmap2
    """
    vis = image.copy()

    contours1, _ = cv2.findContours(mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours2, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # heatmap1 用红色
    cv2.drawContours(vis, contours1, -1, (0, 0, 255), 2)

    # heatmap2 用青色
    cv2.drawContours(vis, contours2, -1, (255, 255, 0), 2)

    cv2.putText(
        vis,
        "Red: Heatmap1   Cyan: Heatmap2",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    return vis


def get_mask_center(mask):
    """
    计算 mask 的响应中心
    """
    ys, xs = np.where(mask > 0)

    if len(xs) == 0 or len(ys) == 0:
        return None

    center_x = int(np.mean(xs))
    center_y = int(np.mean(ys))

    return center_x, center_y


def draw_shift_arrow(image, mask1, mask2):
    """
    画两个 heatmap 响应中心之间的偏移箭头
    """
    vis = image.copy()

    c1 = get_mask_center(mask1)
    c2 = get_mask_center(mask2)

    if c1 is None or c2 is None:
        print("无法计算响应中心")
        return vis, None

    # heatmap1 center: red
    cv2.circle(vis, c1, 8, (0, 0, 255), -1)

    # heatmap2 center: cyan
    cv2.circle(vis, c2, 8, (255, 255, 0), -1)

    # 从 heatmap1 指向 heatmap2
    cv2.arrowedLine(
        vis,
        c1,
        c2,
        (0, 255, 255),
        3,
        tipLength=0.25
    )

    dx = c2[0] - c1[0]
    dy = c2[1] - c1[1]
    distance = np.sqrt(dx ** 2 + dy ** 2)

    text = f"Shift: dx={dx}, dy={dy}, dist={distance:.2f}px"

    cv2.putText(
        vis,
        text,
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    return vis, (dx, dy, distance)


def make_difference_overlay(image, response1, response2, alpha=0.5):
    """
    制作两张 heatmap 的差异图，并 overlay 到原图
    差异越大，说明两个 heatmap 响应越不一致
    """
    diff = cv2.absdiff(response1, response2)
    diff = normalize_to_uint8(diff)

    diff_heatmap = cv2.applyColorMap(diff, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(
        image,
        1 - alpha,
        diff_heatmap,
        alpha,
        0
    )

    cv2.putText(
        overlay,
        "Difference Map: larger response means stronger misalignment",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    return diff, diff_heatmap, overlay


def fuse_heatmaps(heatmap1, heatmap2, mode="weighted", w1=0.5, w2=0.5):
    """
    融合两张彩色 heatmap
    """
    heatmap1 = heatmap1.astype(np.float32)
    heatmap2 = heatmap2.astype(np.float32)

    if mode == "weighted":
        fused = w1 * heatmap1 + w2 * heatmap2
    elif mode == "max":
        fused = np.maximum(heatmap1, heatmap2)
    elif mode == "add":
        fused = heatmap1 + heatmap2
    else:
        raise ValueError("mode 必须是 weighted / max / add")

    fused = np.clip(fused, 0, 255).astype(np.uint8)
    return fused


def overlay_heatmap_on_image(image, heatmap, alpha=0.45):
    overlay = cv2.addWeighted(
        image.astype(np.uint8),
        1 - alpha,
        heatmap.astype(np.uint8),
        alpha,
        0
    )
    return overlay


def add_title(img, title):
    """
    给图像添加标题，方便论文图排版
    """
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 45), (0, 0, 0), -1)
    cv2.putText(
        out,
        title,
        (15, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )
    return out


def concat_panel(images):
    """
    横向拼接多张图
    """
    return np.concatenate(images, axis=1)


def main():
    # =========================
    # 修改为你的路径
    # =========================
    heatmap1_path = r"I:/METU_VisTIR_sunny1-geosam-feature-vis/0005_IM_10308_IM_10308/image0_optical_01_dino_semantic_l4_heatmap.png"
    heatmap2_path = r"I:/METU_VisTIR_sunny1-geosam-feature-vis/0005_IM_10308_IM_10308/image0_optical_03_cnn_guide_1_2_heatmap.png"
    image_path = r"I:/METU_VisTIR/sunny/scene_1/visible/images/IM_10308.jpg"

    save_dir = r"I:/introdction_results"
    ensure_dir(save_dir)

    # =========================
    # 1. 读取图像
    # =========================
    image = read_image(image_path)
    heatmap1 = read_image(heatmap1_path)
    heatmap2 = read_image(heatmap2_path)

    print("原图尺寸:", image.shape)
    print("heatmap1 尺寸:", heatmap1.shape)
    print("heatmap2 尺寸:", heatmap2.shape)

    # =========================
    # 2. 将 heatmap resize 到原图大小
    # =========================
    heatmap1_resize = resize_to_target(heatmap1, image.shape)
    heatmap2_resize = resize_to_target(heatmap2, image.shape)

    # =========================
    # 3. 将已有 heatmap 转成响应强度图
    # =========================
    response1 = heatmap_to_response(heatmap1_resize)
    response2 = heatmap_to_response(heatmap2_resize)

    # =========================
    # 4. 提取高响应区域
    # top_percent 越小，区域越集中
    # 推荐 15 或 20
    # =========================
    mask1 = get_top_response_mask(response1, top_percent=20)
    mask2 = get_top_response_mask(response2, top_percent=20)

    # =========================
    # 5. 画红色/青色轮廓，体现错位
    # =========================
    contour_misalignment = draw_contours_on_image(image, mask1, mask2)

    # =========================
    # 6. 画响应中心偏移箭头
    # =========================
    arrow_misalignment, shift_info = draw_shift_arrow(image, mask1, mask2)

    if shift_info is not None:
        dx, dy, dist = shift_info
        print(f"特征响应中心偏移: dx={dx}, dy={dy}, distance={dist:.2f}px")

    # =========================
    # 7. 计算差异图，体现响应不一致区域
    # =========================
    diff_gray, diff_heatmap, diff_overlay = make_difference_overlay(
        image,
        response1,
        response2,
        alpha=0.5
    )

    # =========================
    # 8. 融合两张 heatmap 后 overlay 到原图
    # 这是你原本想要的 fused heatmap overlay
    # =========================
    fused_heatmap = fuse_heatmaps(
        heatmap1_resize,
        heatmap2_resize,
        mode="weighted",
        w1=0.5,
        w2=0.5
    )

    fused_overlay = overlay_heatmap_on_image(
        image,
        fused_heatmap,
        alpha=0.45
    )

    # =========================
    # 9. 保存所有结果
    # =========================
    cv2.imwrite(os.path.join(save_dir, "heatmap1_resize.png"), heatmap1_resize)
    cv2.imwrite(os.path.join(save_dir, "heatmap2_resize.png"), heatmap2_resize)

    cv2.imwrite(os.path.join(save_dir, "response1_gray.png"), response1)
    cv2.imwrite(os.path.join(save_dir, "response2_gray.png"), response2)

    cv2.imwrite(os.path.join(save_dir, "mask1_top_response.png"), mask1)
    cv2.imwrite(os.path.join(save_dir, "mask2_top_response.png"), mask2)

    cv2.imwrite(os.path.join(save_dir, "misalignment_contours.png"), contour_misalignment)
    cv2.imwrite(os.path.join(save_dir, "misalignment_arrow.png"), arrow_misalignment)

    cv2.imwrite(os.path.join(save_dir, "difference_gray.png"), diff_gray)
    cv2.imwrite(os.path.join(save_dir, "difference_heatmap.png"), diff_heatmap)
    cv2.imwrite(os.path.join(save_dir, "difference_overlay.png"), diff_overlay)

    cv2.imwrite(os.path.join(save_dir, "fused_heatmap.png"), fused_heatmap)
    cv2.imwrite(os.path.join(save_dir, "fused_overlay_on_image.png"), fused_overlay)

    # =========================
    # 10. 生成论文展示用 panel
    # =========================
    image_panel = add_title(image, "Original Image")
    contour_panel = add_title(contour_misalignment, "Misalignment Contours")
    diff_panel = add_title(diff_overlay, "Difference Overlay")
    fused_panel = add_title(fused_overlay, "Fused Heatmap Overlay")

    paper_panel = concat_panel([
        image_panel,
        contour_panel,
        diff_panel,
        fused_panel
    ])

    cv2.imwrite(os.path.join(save_dir, "paper_misalignment_panel.png"), paper_panel)

    print("处理完成，结果保存到:", save_dir)
    print("重点查看这几张：")
    print("1. misalignment_contours.png      -> 红色/青色轮廓错位")
    print("2. misalignment_arrow.png         -> 响应中心偏移箭头")
    print("3. difference_overlay.png         -> heatmap 差异叠加图")
    print("4. fused_overlay_on_image.png     -> 融合 heatmap 叠加原图")
    print("5. paper_misalignment_panel.png   -> 论文展示拼图")


if __name__ == "__main__":
    main()