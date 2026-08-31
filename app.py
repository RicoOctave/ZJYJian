import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import cv2
import math
import csv
import logging
import sys
from scipy.interpolate import splprep, splev
from skimage.morphology import skeletonize
from skimage import img_as_bool
from io import BytesIO
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# ===================== 【工程配置：全局常量集中管理｜答辩、调参直接修改此处，不侵入业务逻辑】 =====================
class ProjectConfig:
    """项目全局常量配置类，所有阈值、权重、超参统一在此维护，便于论文调参、复现实验"""
    # 加权模型权重
    W1_STROKE: float = 0.20
    W2_BEZIER: float = 0.25
    W3_ENTROPY: float = 0.35
    W4_RESERVED: float = 0.20

    # 判别阈值
    AI_THRESHOLD: float = 0.42
    HUMAN_THRESHOLD: float = 0.62

    # 算法超参
    K3M_ITER_STAGES: int = 7
    DOUGLAS_EPSILON: float = 1.5
    BEZIER_CONTROL_POINTS: int = 8
    MIN_STROKE_LENGTH: float = 20.0
    MIN_SPEED_VALID_COUNT: int = 2
    DIST_TRANSFORM_KERNEL: int = 3

    # 形态学预处理
    MORPH_RECT_W: int = 2
    MORPH_RECT_H: int = 2

    # 数值安全下限
    EPS_FLOAT: float = 1e-6

# 日志配置，工程级日志输出，区分INFO/WARNING/ERROR
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# ======================== 【对齐申报书附件完整算法，增加类型注解、入参校验、日志容错，核心数学逻辑完全不变】 ========================
def cv2_imread_chinese(path: str, flag: int = 0) -> Optional[np.ndarray]:
    """
    申报书附件原版：支持中文路径、中文文件名读取图片（网页内存上传不调用，本地批量脚本可用）
    Args:
        path: 图像文件路径
        flag: 0=灰度图，1=彩色BGR图
    Returns:
        np.ndarray图像矩阵，读取失败返回None
    """
    if not isinstance(path, str) or len(path.strip()) == 0:
        logger.error("cv2_imread_chinese: 文件路径参数非法")
        return None
    try:
        with open(path, 'rb') as stream:
            bytes_data = bytearray(stream.read())
            np_array = np.asarray(bytes_data, dtype=np.uint8)
            img = cv2.imdecode(np_array, flag)
        return img
    except Exception as e:
        logger.error(f"cv2_imread_chinese 读取图片失败：path={path}, err={e}")
        return None


class K3MSkeletonExtractor:
    """
    K3M七阶段迭代骨架提取器
    实现：查找表预构建、8邻域编码、小连通域噪声过滤、笔画断点修复
    """
    def __init__(self):
        self.lookup_table: np.ndarray = self._build_k3m_lookup()

    def _build_k3m_lookup(self) -> np.ndarray:
        lookup = np.zeros(512, dtype=np.bool_)
        for i in range(512):
            bin_str = np.binary_repr(i, 9)
            center = int(bin_str[4])
            if center == 0:
                lookup[i] = False
                continue
            neighbors = [int(bin_str[j]) for j in [0,1,2,3,5,6,7,8]]
            n_count = sum(neighbors)
            transitions = 0
            for j in range(8):
                transitions += neighbors[j] and not neighbors[(j+1)%8]
            lookup[i] = (n_count >= 2 and n_count <= 6) and (transitions == 1)
        return lookup

    def _get_neighborhood_code(self, img: np.ndarray, x: int, y: int) -> int:
        h, w = img.shape
        code = 0
        idx = 0
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                ny, nx = y + dy, x + dx
                val = 1 if (0 <= ny < h and 0 <= nx < w and img[ny, nx] == 255) else 0
                code += val * (2 ** idx)
                idx += 1
        return code

    def _adaptive_filter(self, skeleton: np.ndarray) -> np.ndarray:
        h, w = skeleton.shape
        filtered = skeleton.copy()
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(skeleton, connectivity=8)
        for label in range(1, num_labels):
            if stats[label, cv2.CC_STAT_AREA] < 3:
                filtered[labels == label] = 0
        return filtered

    def _repair_breakpoints(self, skeleton: np.ndarray) -> np.ndarray:
        h, w = skeleton.shape
        repaired = skeleton.copy()
        for y in range(1, h-1):
            for x in range(1, w-1):
                if repaired[y, x] == 0:
                    neighbors = []
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            if dx == 0 and dy == 0:
                                continue
                            ny, nx = y + dy, x + dx
                            if repaired[ny, nx] == 255:
                                neighbors.append((ny, nx))
                    if len(neighbors) == 2:
                        repaired[y, x] = 255
        return repaired

    def extract_skeleton(self, binary_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        K3M执行骨架提取
        Args:
            binary_img: 单通道二值图像，前景笔迹=255，背景=0
        Returns:
            skeleton: 骨架图像矩阵
            distance: 距离变换矩阵，用于反推笔画宽度
        """
        if len(binary_img.shape) != 2:
            raise ValueError("extract_skeleton:输入必须为单通道二值图像")
        img = binary_img.copy()
        h, w = img.shape
        skeleton = np.zeros_like(img, dtype=np.uint8)
        distance = np.zeros_like(img, dtype=np.float32)
        for _ in range(ProjectConfig.K3M_ITER_STAGES):
            mask = np.zeros_like(img, dtype=np.bool_)
            for y in range(1, h-1):
                for x in range(1, w-1):
                    if img[y, x] == 255:
                        code = self._get_neighborhood_code(img, x, y)
                        if self.lookup_table[code]:
                            mask[y, x] = True
            img[mask] = 0
        skeleton[img == 255] = 255
        skeleton = self._adaptive_filter(skeleton)
        skeleton = self._repair_breakpoints(skeleton)
        dist = cv2.distanceTransform(binary_img, cv2.DIST_L2, ProjectConfig.DIST_TRANSFORM_KERNEL)
        distance[skeleton == 255] = dist[skeleton == 255]
        return skeleton, distance


def calculate_stroke_cv(binary_img: np.ndarray) -> Dict[str, Any]:
    """
    【申报书原版完整函数，返回全部中间实验参数】笔画粗细变异系数计算
    Args:
        binary_img:二值单通道笔迹图像
    Returns:
        dict: stroke_cv笔画变异系数, mean_width平均宽度, std_width标准差, skeleton骨架图
    """
    k3m_extractor = K3MSkeletonExtractor()
    skeleton, distance = k3m_extractor.extract_skeleton(binary_img)
    skeleton_points = skeleton == 255
    widths = distance[skeleton_points] * 2
    widths = widths[widths > ProjectConfig.EPS_FLOAT]
    if len(widths) == 0:
        logger.warning("calculate_stroke_cv: 当前图像未检测到有效笔迹，特征置0")
        return {
            "stroke_cv":0.0,
            "mean_width":0.0,
            "std_width":0.0,
            "skeleton":skeleton
        }
    mean_width = np.mean(widths)
    std_width = np.std(widths, ddof=1)
    stroke_cv = std_width / mean_width if mean_width > ProjectConfig.EPS_FLOAT else 0.0
    return {
        "stroke_cv": round(stroke_cv,4),
        "mean_width": round(mean_width,2),
        "std_width": round(std_width,2),
        "skeleton": skeleton
    }


def calculate_bezier_residual(binary_img: np.ndarray, num_control_points: int =None) -> Dict[str, Any]:
    """
    【申报书原版完整贝塞尔，返回均值残差、最大残差、有效笔画长度】
    Args:
        binary_img:二值笔迹图像
        num_control_points:贝塞尔控制点数量，默认读取全局配置
    Returns:
        dict: mean_res平均残差，max_res最大残差，total_len有效笔画总长度
    """
    if num_control_points is None:
        num_control_points = ProjectConfig.BEZIER_CONTROL_POINTS
    bool_img = img_as_bool(binary_img)
    skeleton = skeletonize(bool_img)
    skeleton_8u = (skeleton * 255).astype(np.uint8)
    contours, _ = cv2.findContours(skeleton_8u, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours)==0:
        logger.warning("calculate_bezier_residual:未找到笔迹轮廓")
        return {"mean_res":0.0,"max_res":0.0,"total_len":0.0}
    valid_contours:List[Tuple[np.ndarray,float]]=[]
    total_valid_length=0.0
    for contour in contours:
        coords = contour.squeeze()
        if len(coords)<2 or coords.ndim !=2:
            continue
        contour_length=0.0
        for i in range(1,len(coords)):
            dx=coords[i,0]-coords[i-1,0]
            dy=coords[i,1]-coords[i-1,1]
            contour_length+=math.hypot(dx,dy)
        if contour_length>=ProjectConfig.MIN_STROKE_LENGTH:
            valid_contours.append((coords,contour_length))
            total_valid_length+=contour_length
    if len(valid_contours)==0 or total_valid_length<ProjectConfig.MIN_STROKE_LENGTH:
        return {"mean_res":0.0,"max_res":0.0,"total_len":round(total_valid_length,2)}
    all_residuals:List[float]=[]
    for coords,_ in valid_contours:
        if len(coords) < num_control_points+1:
            continue
        try:
            tck,u = splprep(coords.T, s=0, k=3, nest=num_control_points)
            u_new = np.linspace(0,1,len(coords))
            fitted_points = np.array(splev(u_new, tck)).T
            for i in range(len(coords)):
                dx=coords[i,0]-fitted_points[i,0]
                dy=coords[i,1]-fitted_points[i,1]
                residual=math.hypot(dx,dy)
                all_residuals.append(residual)
        except Exception as e:
            logger.debug(f"贝塞尔单条轮廓拟合异常:{str(e)}")
            continue
    if len(all_residuals)==0:
        return {"mean_res":0.0,"max_res":0.0,"total_len":round(total_valid_length,2)}
    mean_residual = np.mean(all_residuals)
    max_residual = np.max(all_residuals)
    return {
        "mean_res":round(mean_residual,4),
        "max_res":round(max_residual,4),
        "total_len":round(total_valid_length,2)
    }


def douglas_peucker(points:np.ndarray, epsilon: Optional[float]=None) -> np.ndarray:
    """
    道格拉斯‑普克关键点采样算法，提取笔迹骨架特征点
    Args:
        points:Nx2坐标点数组
        epsilon:压缩阈值，读取全局配置
    Returns:
        压缩后的关键点数组
    """
    if epsilon is None:
        epsilon = ProjectConfig.DOUGLAS_EPSILON
    if len(points) <3:
        return points.copy()
    start_point = points[0]
    end_point = points[-1]
    max_dist=0.0
    max_idx=0
    for i in range(1,len(points)-1):
        line_vec = end_point - start_point
        point_vec = points[i] - start_point
        line_len = np.linalg.norm(line_vec)
        if line_len < ProjectConfig.EPS_FLOAT:
            dist = np.linalg.norm(point_vec)
        else:
            proj_len = np.dot(point_vec, line_vec)/line_len
            proj_len = np.clip(proj_len,0,line_len)
            proj_point = start_point + proj_len * line_vec / line_len
            dist = np.linalg.norm(points[i]-proj_point)
        if dist>max_dist:
            max_dist=dist
            max_idx=i
    if max_dist>epsilon:
        left_points = douglas_peucker(points[:max_idx+1],epsilon)
        right_points = douglas_peucker(points[max_idx:],epsilon)
        return np.vstack((left_points[:-1], right_points))
    else:
        return np.array([start_point, end_point])


def calculate_speed_entropy(binary_img: np.ndarray, epsilon: Optional[float]=None) -> Dict[str, Any]:
    """
    【完全对齐申报书附件原版：返回全套中间实验指标】书写等效波动熵计算
    Args:
        binary_img:二值笔迹图像
        epsilon:道格拉斯‑普克阈值
    Returns:
        speed_cv, feature_density, speed_mean, speed_std, entropy, valid_len
    """
    if epsilon is None:
        epsilon = ProjectConfig.DOUGLAS_EPSILON
    bool_img = img_as_bool(binary_img)
    skeleton = skeletonize(bool_img)
    skeleton_8u = (skeleton*255).astype(np.uint8)
    contours,_ = cv2.findContours(skeleton_8u, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours)==0:
        logger.warning("calculate_speed_entropy:未检测笔迹轮廓")
        return {
            "speed_cv":0.0,
            "feature_density":0.0,
            "speed_mean":0.0,
            "speed_std":0.0,
            "entropy":0.0,
            "valid_len":0.0
        }
    all_equivalent_speed:List[float]=[]
    all_original_points:List[np.ndarray]=[]
    all_feature_points:List[np.ndarray]=[]
    total_valid_length=0.0
    for contour in contours:
        original_points = contour.squeeze()
        if len(original_points)<2 or original_points.ndim !=2:
            continue
        contour_length=0.0
        for i in range(1,len(original_points)):
            contour_length+=np.linalg.norm(original_points[i]-original_points[i-1])
        if contour_length<ProjectConfig.MIN_STROKE_LENGTH:
            continue
        total_valid_length+=contour_length
        all_original_points.extend(original_points.tolist())
        feature_points = douglas_peucker(original_points,epsilon)
        if len(feature_points)<2:
            continue
        all_feature_points.extend(feature_points.tolist())
        for i in range(1,len(feature_points)):
            speed = np.linalg.norm(feature_points[i]-feature_points[i-1])
            if speed>ProjectConfig.EPS_FLOAT:
                all_equivalent_speed.append(float(speed))
    if len(all_equivalent_speed)<ProjectConfig.MIN_SPEED_VALID_COUNT:
        fd = len(all_feature_points)/len(all_original_points) if len(all_original_points)>0 else 0.0
        return {
            "speed_cv":0.0,
            "feature_density":round(fd,4),
            "speed_mean":0.0,
            "speed_std":0.0,
            "entropy":0.0,
            "valid_len":round(total_valid_length,2)
        }
    speed_array = np.array(all_equivalent_speed)
    speed_mean = np.mean(speed_array)
    speed_std = np.std(speed_array, ddof=1)
    speed_cv = speed_std/speed_mean if abs(speed_mean) > ProjectConfig.EPS_FLOAT else 0.0
    acc_list:List[float]=[]
    for i in range(1,len(all_equivalent_speed)):
        acc_list.append(abs(all_equivalent_speed[i]-all_equivalent_speed[i-1]))
    acc_array=np.array(acc_list)
    entropy=0.0
    if len(acc_array)>0 and np.max(acc_array)>ProjectConfig.EPS_FLOAT:
        acc_norm = acc_array / np.max(acc_array)
        hist,_ = np.histogram(acc_norm,bins=10,density=True)
        hist = hist[hist>ProjectConfig.EPS_FLOAT]
        entropy = -np.sum(hist * np.log2(hist)) / np.log2(len(hist))
    fd = len(all_feature_points)/len(all_original_points) if len(all_original_points)>0 else 0.0
    return {
        "speed_cv":round(speed_cv,4),
        "feature_density":round(fd,4),
        "speed_mean":round(speed_mean,4),
        "speed_std":round(speed_std,4),
        "entropy":round(entropy,4),
        "valid_len":round(total_valid_length,2)
    }


def weighted_model_detect(cv_stroke: float, bezier_mean: float, entropy_val: float) -> Dict[str, Any]:
    """
    多特征加权判别模型，权重、归一化、阈值完全对齐申报书
    Args:
        cv_stroke:笔画粗细变异系数
        bezier_mean:贝塞尔平均拟合残差
        entropy_val:书写等效波动熵
    Returns:
        dict: score综合得分, result文本结果, is_ai标记True/False/None
    """
    w1,w2,w3,w4 = ProjectConfig.W1_STROKE, ProjectConfig.W2_BEZIER, ProjectConfig.W3_ENTROPY, ProjectConfig.W4_RESERVED
    norm_cv = np.clip((cv_stroke - 0.46)/(0.72-0.46),0,1)
    norm_bezier = np.clip(bezier_mean /70.0,0,1)
    norm_entropy = np.clip((entropy_val +4.8)/(0.9),0,1)
    S = w1*norm_cv + w2*norm_bezier + w3*norm_entropy
    ai_threshold = ProjectConfig.AI_THRESHOLD
    human_threshold = ProjectConfig.HUMAN_THRESHOLD
    if S < ai_threshold:
        return {"score":round(S,4),"result":"判定：高度疑似AI伪造电子签名","is_ai":True}
    elif S> human_threshold:
        return {"score":round(S,4),"result":"判定：高度疑似真人书写电子签名","is_ai":False}
    else:
        return {"score":round(S,4),"result":"判定：处于过渡区间，建议人工复核","is_ai":None}


def single_image_full_pipeline(gray_img:np.ndarray) -> Dict[str, Any]:
    """
    【工程封装：单张图片完整流水线入口函数】
    输入灰度图像，完成二值化、形态学、三大特征计算、加权判别，对外统一接口
    Args:
        gray_img:单通道灰度图像
    Returns:
        完整结果字典：binary_img, stroke, bezier, speed, model
    """
    _, binary = cv2.threshold(gray_img, 0,255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(ProjectConfig.MORPH_RECT_W, ProjectConfig.MORPH_RECT_H))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    stroke_res = calculate_stroke_cv(binary)
    bezier_res = calculate_bezier_residual(binary)
    speed_res = calculate_speed_entropy(binary)
    model_res = weighted_model_detect(stroke_res["stroke_cv"], bezier_res["mean_res"], speed_res["entropy"])
    return {
        "binary_img":binary,
        "stroke":stroke_res,
        "bezier":bezier_res,
        "speed":speed_res,
        "model":model_res
    }


def batch_folder_analysis(folder_path:str) -> Dict[str, Dict[str, Any]]:
    """
    本地文件夹批量完整分析（CLI模式科研实验用）
    Args:
        folder_path:签名图片文件夹路径
    Returns:
        key:文件名 value:全套特征与判别结果
    """
    supported_formats = (".png",".jpg",".jpeg",".bmp",".tiff")
    results:Dict[str,Dict[str,Any]] = {}
    if not os.path.isdir(folder_path):
        raise NotADirectoryError(f"文件夹不存在 {folder_path}")
    file_list = os.listdir(folder_path)
    logger.info(f"批量分析启动，扫描文件总数：{len(file_list)}")
    for filename in os.listdir(folder_path):
        if not filename.lower().endswith(supported_formats):
            continue
        full_path = os.path.join(folder_path, filename)
        logger.info(f"处理样本：{filename}")
        img_gray = cv2_imread_chinese(full_path, flag=0)
        if img_gray is None:
            logger.warning(f"跳过读取失败文件 {filename}")
            continue
        try:
            res_all = single_image_full_pipeline(img_gray)
            results[filename] = {
                "stroke_cv":res_all["stroke"]["stroke_cv"],
                "mean_width":res_all["stroke"]["mean_width"],
                "std_width":res_all["stroke"]["std_width"],
                "bezier_mean_res":res_all["bezier"]["mean_res"],
                "bezier_max_res":res_all["bezier"]["max_res"],
                "bezier_total_len":res_all["bezier"]["total_len"],
                "speed_cv":res_all["speed"]["speed_cv"],
                "feature_density":res_all["speed"]["feature_density"],
                "speed_mean":res_all["speed"]["speed_mean"],
                "speed_std":res_all["speed"]["speed_std"],
                "entropy":res_all["speed"]["entropy"],
                "speed_valid_len":res_all["speed"]["valid_len"],
                "score_S":res_all["model"]["score"],
                "is_ai":res_all["model"]["is_ai"],
                "judge_text":res_all["model"]["result"]
            }
        except Exception as e:
            logger.error(f"样本计算异常 {filename}, err={str(e)}")
            continue
    logger.info(f"批量处理结束，有效样本数量 {len(results)}")
    return results


def export_batch_csv(result_dict:Dict[str,Dict[str,Any]], out_csv_path:str):
    """批量结果导出CSV，SPSS/Pandas统计分析直接读取"""
    headers = [
        "文件名","stroke_cv笔画变异系数","mean_width平均笔画宽度","std_width笔画标准差",
        "bezier_mean_res贝塞尔平均残差","bezier_max_res最大残差","bezier_total_len有效笔画长度",
        "speed_cv等效速度变异系数","feature_density特征点密度","speed_mean等效速度均值",
        "speed_std等效速度标准差","entropy等效波动熵","speed_valid_len书写有效长度",
        "score_S综合得分","is_ai是否AI(None/True/False)","judge_text判别文字"
    ]
    with open(out_csv_path,"w",encoding="utf-8-sig",newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for fname,item in result_dict.items():
            row = {"文件名":fname,**item}
            writer.writerow(row)
    logger.info(f"CSV批量报告已输出至 {out_csv_path}")


def run_cli_console():
    """本地命令行交互控制台，用于科研批量实验"""
    print("="*75)
    print("📝真迹云鉴｜底层算法控制台")
    print("1:单张图片完整分析 | 2:文件夹批量分析导出CSV | q:退出")
    print("="*75)
    while True:
        sel = input("\n请输入功能选项：").strip()
        if sel.lower() == "q":
            print("\n程序退出")
            break
        elif sel == "1":
            p = input("输入图片完整路径：").strip()
            img_g = cv2_imread_chinese(p,0)
            if img_g is None:
                print("图片读取失败")
                continue
            res = single_image_full_pipeline(img_g)
            print("\n--------单样本全部结果--------")
            print(f"笔画模块 cv={res['stroke']['stroke_cv']} mean_w={res['stroke']['mean_width']}")
            print(f"贝塞尔模块 mean_res={res['bezier']['mean_res']}")
            print(f"书写熵模块 entropy={res['speed']['entropy']}")
            print(f"判别结果 S={res['model']['score']} → {res['model']['result']}")
        elif sel == "2":
            folder_in = input("输入图片文件夹路径：").strip()
            try:
                batch_result = batch_folder_analysis(folder_in)
                csv_save = input("输入输出CSV文件路径(回车跳过不导出)：").strip()
                if len(csv_save.strip())>0:
                    export_batch_csv(batch_result, csv_save)
                for fn,val in batch_result.items():
                    print(f"\n[{fn}] score_S={val['score_S']} judge={val['judge_text']}")
            except Exception as e:
                print(f"批量运行异常：{str(e)}")
        else:
            print("输入选项无效，请重新选择")


def generate_pdf_report(full_data:Dict[str,Any]) -> BytesIO:
    """PDF报告扩充：写入全套中间实验参数，满足科研实验留存"""
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    width, height = A4
    c.setFont("Helvetica-Bold",16)
    c.drawCentredString(width/2, height-50,"真迹云鉴 AI伪造电子签名教学筛查报告")
    c.setFont("Helvetica",11)
    c.drawString(40, height-80,"项目来源：北京市级大学生创新训练项目")
    c.drawString(40, height-95,"产品主体：真迹云鉴（佛山市）科技有限公司")
    c.drawString(40, height-110,"⚠声明：本报告仅用于科研教学演示，不具备司法鉴定法律效力。")
    c.line(40,height-125,width-40,height-125)
    y = height-150

    c.drawString(40,y,"=== 判别结论 ===")
    y -=22
    c.drawString(40,y,"判定结果：%s"%full_data["model_result"]["result"])
    y -=22
    c.drawString(40,y,"综合加权得分S：%s"%full_data["model_result"]["score"])
    y -=30

    c.drawString(40,y,"=== 笔画粗细变异系数模块（K3M骨架） ===")
    y -=22
    c.drawString(40,y,"笔画粗细变异系数：%s"%full_data["stroke"]["stroke_cv"])
    y -=22
    c.drawString(40,y,"笔画平均宽度(像素)：%s"%full_data["stroke"]["mean_width"])
    y -=22
    c.drawString(40,y,"笔画宽度标准差(像素)：%s"%full_data["stroke"]["std_width"])
    y -=30

    c.drawString(40,y,"=== 贝塞尔曲线拟合残差模块 ===")
    y -=22
    c.drawString(40,y,"平均拟合残差(像素)：%s"%full_data["bezier"]["mean_res"])
    y -=22
    c.drawString(40,y,"最大拟合残差(像素)：%s"%full_data["bezier"]["max_res"])
    y -=22
    c.drawString(40,y,"有效笔画总长度(像素)：%s"%full_data["bezier"]["total_len"])
    y -=30

    c.drawString(40,y,"=== 书写速度等效波动熵模块（道格拉斯‑普克） ===")
    y -=22
    c.drawString(40,y,"书写速度等效变异系数：%s"%full_data["speed"]["speed_cv"])
    y -=22
    c.drawString(40,y,"特征点密度：%s"%full_data["speed"]["feature_density"])
    y -=22
    c.drawString(40,y,"等效速度均值(像素)：%s"%full_data["speed"]["speed_mean"])
    y -=22
    c.drawString(40,y,"等效速度标准差(像素)：%s"%full_data["speed"]["speed_std"])
    y -=22
    c.drawString(40,y,"书写等效波动熵：%s"%full_data["speed"]["entropy"])
    y -=22
    c.drawString(40,y,"有效笔画总长度(像素)：%s"%full_data["speed"]["valid_len"])

    c.showPage()
    c.save()
    buf.seek(0)
    return buf

# ======================== Streamlit页面开始 【美化增强｜算法完整对齐申报书附件】 ========================
st.set_page_config(
    page_title="真迹云鉴 · AI伪造签名快筛系统",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 自定义CSS 高级深蓝科技风
st.markdown("""
<style>
.main {
    background-color:#f3f7fb;
}
.block-container {
    padding-top:2rem;
    padding-bottom:3rem;
    max-width:1400px;
}
div[data-testid="stVerticalBlockBorderWrapper"]{
    border-radius:12px !important;
    box-shadow: 0 2px 14px rgba(15, 50, 110, 0.12) !important;
    border:1px solid #e2ebf6 !important;
}
.stTabs [data-baseweb="tab-list"] {
	gap:8px;
}
.stTabs [data-baseweb="tab"] {
    height:44px;
    border-radius:8px 8px 0 0;
	padding:0px 20px;
	font-weight:500;
}
.stTabs [aria-selected="true"] {
    background-color:#164b96 !important;
    color:#ffffff !important;
}
h1,h2,h3,h4 {
    color:#0f3460 !important;
}
div[data-testid="stInfo"] {
    border-left:6px solid #2176d2 !important;
}
div[data-testid="stSuccess"] {
    border-left:6px solid #00875a !important;
}
div[data-testid="stWarning"] {
    border-left:6px solid #ff8b00 !important;
}
div[data-testid="stError"] {
    border-left:6px solid #d92d20 !important;
}
[data-testid="stFileUploader"]{
    border:2px dashed #94b8e0;
    border-radius:10px;
    padding:16px;
    background:#f8fbff;
}
hr {
    border-top:1px solid #c9d8ec !important;
}
.footer-note{
    font-size:0.85rem;
    color:#546b8c;
    text-align:center;
    margin-top:40px;
}
</style>
""",unsafe_allow_html=True)

# ==========页面头部大标题区域==========
st.markdown("""
<div style="background:linear-gradient(90deg,#082248,#0f3b7c);padding:24px 28px;border-radius:12px;color:#ffffff;margin-bottom:24px;">
<h2 style="color:#FFFFFF !important;margin:0;font-weight:900;font-size:1.9rem;letter-spacing:0.8px;text-shadow:0px 3px 3px #000000, 0px 5px 10px rgba(0,0,0,0.98);-webkit-text-stroke:0.8px rgba(0,0,0,0.75);">✍️ 真迹云鉴 — AI伪造电子签名轻量化快筛系统</h2>
<p style="opacity:0.95;margin:12px 0 0 0;color:#ffffff;">北京市级大学生创新训练项目｜V8.1 
&nbsp;&nbsp;
<span style="background:#ffffff;color:#0f3b7c;padding:4px 10px;border-radius:16px;font-size:0.85rem;">科研教学演示版本</span>
</p>
<p style="opacity:0.9;font-size:0.9rem;margin:6px 0 0 0;color:#ffffff;">⚠本系统仅用于高校科研、课堂实训演示，不具备司法鉴定法律效力</p>
<p style="opacity:0.75;font-size:0.82rem;margin:10px 0 0 0;color:#ffffff;">产品所属主体：真迹云鉴（佛山市）科技有限公司</p>
</div>
""", unsafe_allow_html=True)

tab_detect, tab_teach = st.tabs(["🧪 签名检测模块（实操检测）","📖 高校教学教具演示模块（课堂教学）"])

# --------------------------- 模块一：签名检测模块【完整算法输出｜全部中间实验指标展示】 ---------------------------
with tab_detect:
    st.markdown('<h3>AI伪造签名筛查工具</h3>',unsafe_allow_html=True)

    col_info = st.container(border=True)
with col_info:
    feature_weight_html = """
特征权重分配：
<span style='background:#164b96;color:white;padding:2px 8px;border-radius:8px;font-size:0.8rem;'>笔画粗细变异系数 20%</span>
<span style='background:#164b96;color:white;padding:2px 8px;border-radius:8px;font-size:0.8rem;'>贝塞尔曲线拟合残差 25%</span>
<span style='background:#164b96;color:white;padding:2px 8px;border-radius:8px;font-size:0.8rem;'>书写速度等效波动熵 35%</span>
<span style='background:#164b96;color:white;padding:2px 8px;border-radius:8px;font-size:0.8rem;'>异常部件发生率 20%</span>
"""
    st.markdown(feature_weight_html, unsafe_allow_html=True)
    st.info("💡注意：")
    uploaded_file = st.file_uploader(
        "📤 上传签名图片，优先白底黑字签名，支持JPG / PNG格式",
        type=["jpg","jpeg","png"]
    )

    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        pil_img = Image.open(BytesIO(bytes_data)).convert("L")
        img_gray = np.array(pil_img)

        with st.spinner("🔍算法运算中：二值化、K3M骨架提取、贝塞尔拟合、道格拉斯‑普克特征点提取、等效波动熵计算、加权模型判别……"):
            pipeline_result = single_image_full_pipeline(img_gray)
            stroke_result = pipeline_result["stroke"]
            bezier_result = pipeline_result["bezier"]
            speed_result = pipeline_result["speed"]
            model_out = pipeline_result["model"]
            binary = pipeline_result["binary_img"]

        st.success("✅算法计算完成，展示算法全流程中间输出产物")

        col_img1, col_img2, col_img3 = st.columns(3)
        with col_img1:
            c1 = st.container(border=True)
            with c1:
                st.subheader("原始签名图像")
                st.image(pil_img,width=300)
        with col_img2:
            c2 = st.container(border=True)
            with c2:
                st.subheader("二值化预处理图像")
                st.image(binary,width=300)
        with col_img3:
            c3 = st.container(border=True)
            with c3:
                st.subheader("K3M算法笔迹骨架")
                st.image(stroke_result["skeleton"],width=300)

        st.divider()

        # ======判别结果卡片 ======
        res_container = st.container(border=True)
        with res_container:
            st.subheader("📌判别结果 & 综合加权得分 S")
            score_S = model_out["score"]
            if model_out["is_ai"] is True:
                risk_text = model_out["result"]
                st.error(f"🔴 {risk_text} ｜综合加权得分 S = {score_S}")
            elif model_out["is_ai"] is False:
                risk_text = model_out["result"]
                st.success(f"🟢 {risk_text} ｜综合加权得分 S = {score_S}")
            else:
                risk_text = model_out["result"]
                st.warning(f"🟡 {risk_text} ｜综合加权得分 S = {score_S}")

        st.divider()
        # =====【完整全套实验参数，分三大模块展示，答辩体现算法专业性】=====
        feat_container = st.container(border=True)
        with feat_container:
            st.subheader("📊全套量化实验参数")
            tab_stroke, tab_bezier, tab_speed = st.tabs(["①笔画粗细模块(K3M)","②贝塞尔残差模块","③书写速度等效熵模块"])
            with tab_stroke:
                sr = stroke_result
                st.markdown(f"""
- **笔画粗细变异系数：`{sr['stroke_cv']}`** ｜AI参考：0.46‑0.54｜真人参考：0.54‑0.72
- **笔画平均宽度(像素)：`{sr['mean_width']}`**
- **笔画宽度标准差(像素)：`{sr['std_width']}`**

> 释义：K3M骨架到笔迹边缘距离反推笔画宽度；AI生成签名笔画分布更均匀，变异系数更低。
""")
            with tab_bezier:
                br = bezier_result
                st.markdown(f"""
- **贝塞尔曲线平均拟合残差(像素)：`{br['mean_res']}`**｜AI参考：<32｜真人参考：>55
- **贝塞尔曲线最大拟合残差(像素)：`{br['max_res']}`**
- **有效笔画总长度(像素)：`{br['total_len']}`**

> 释义：AI伪造签名原生基于贝塞尔曲线生成，拟合残差显著低于真人手写。
""")
            with tab_speed:
                sp = speed_result
                st.markdown(f"""
- **书写速度等效变异系数：`{sp['speed_cv']}`**
- **特征点密度：`{sp['feature_density']}`**
- **等效速度均值(像素)：`{sp['speed_mean']}`**
- **等效速度标准差(像素)：`{sp['speed_std']}`**
- **书写速度等效波动熵：`{sp['entropy']}`**｜AI参考：‑3.8 ~ ‑2.6｜真人参考：‑4.8 ~ ‑3.9
- **有效笔画总长度(像素)：`{sp['valid_len']}`**

> 释义：道格拉斯‑普克算法提取骨架特征点，静态图像等效还原书写运笔节奏；真人受生理抖动熵更低。
""")

        st.divider()
        #风险解读折叠面板
        with st.expander("📖点击展开：风险与结果解读（课堂教学使用）",expanded=False):
            st.markdown("""
1. 如果笔画粗细变异系数偏低，贝塞尔残差偏小，波动熵偏大，综合得分S小于0.42，大概率属于AI模型生成伪造签名；
2. 如果笔画粗细变异系数偏高，贝塞尔残差大，波动熵偏小，综合得分S大于0.62，大概率属于真人手写签名；
3. 0.42‑0.62为灰色过渡区间，受图片清晰度、签名大小、扫描质量干扰，必须人工复核，不能直接下定论；
4. ⚠本模型仅作为**初筛辅助工具，不能直接替代司法鉴定，仅适合高校教学演示、科创项目实验**。
""")

        st.divider()
        #雷达图（保留三个核心判别指标）
        chart_container = st.container(border=True)
        with chart_container:
            st.subheader("📈核心判别指标雷达可视化图表")
            categories = ["笔画粗细变异系数","贝塞尔平均残差","书写等效波动熵"]
            values = [stroke_result["stroke_cv"], bezier_result["mean_res"], speed_result["entropy"]]
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(r=values, theta=categories, fill="toself",name="本次检测样本"))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True)),
                height=420,
                title="样本核心判别指标雷达图",
                template="plotly_white"
            )
            st.plotly_chart(fig_radar,use_container_width=True)

        # PDF传入完整全部数据
        report_full = {
            "model_result":model_out,
            "stroke":stroke_result,
            "bezier":bezier_result,
            "speed":speed_result
        }
        pdf_bytes = generate_pdf_report(report_full)
        st.download_button("📄下载完整PDF实验筛查报告（含全部中间参数）",data=pdf_bytes,file_name="真迹云鉴_完整实验报告.pdf",mime="application/pdf")

    else:
        st.info("👆请上传一张签名图片，开始AI伪造电子签名筛查。")


# --------------------------- 模块二：高校教学教具演示模块【美化增强】 ---------------------------
with tab_teach:
    st.markdown('<h3>高校课堂教学演示教具</h3>',unsafe_allow_html=True)
    st.caption("面向侦查学、刑事司法、证据法学专业课程，用于课堂授课、小组实训、科创项目展示，内容来源于北京市大学生创新训练项目申报书与商业计划书")

    tab_overview, tab_market, tab_tech, tab_exp_chart, tab_business, tab_achievement = st.tabs([
        "📋项目总体概述",
        "📈市场背景与行业痛点",
        "⚙完整核心技术体系",
        "📊实验数据可视化",
        "💼商业模式介绍",
        "🏆项目落地成果与未来展望"
    ])

    # 子tab1：项目总体概述
    with tab_overview:
        box1 = st.container(border=True)
        with box1:
            st.subheader("项目名称：真迹云鉴——AI伪造电子签名的轻量化快筛技术")
            st.markdown("""
### 项目来源
北京市大学生创新训练项目成果转化项目。
随着AI造字技术快速普及，仅仅少量手写样本就可以生成视觉高度仿真的电子签名，肉眼很难分辨真伪，给电子合同、政务文书、司法鉴定带来巨大风险。

### 项目定位
打造**轻量化、低成本、易操作**的AI伪造电子签名快速筛查工具。
- 不需要动态书写轨迹；
- 不需要比对样本；
- 不需要专业硬件设备；
- 仅依靠单张静态签名图片识别AI伪造痕迹。

### 四大目标客户群体
1. 北京地区中小企业
2. 司法鉴定机构
3. 高校（本教具就是面向高校教学实训场景）
4. 个人用户

> ✨本网页算法完全对齐申报书附件独立源码，完整输出全部中间实验指标；学生既可以切换标签页上传签名做实操检测，也可以在这里学习完整的项目原理、技术细节、行业背景。
""")

    # 子tab2：市场背景与行业痛点
    with tab_market:
        box2 = st.container(border=True)
        with box2:
            st.subheader("行业痛点分析")
            st.markdown("""
#### 现存三大痛点
1. **传统司法鉴定门槛高**：单次鉴定收费约3300元，周期3‑7个工作日，成本高、效率低，基层场景无法高频使用；
2. **基层“三无困境”**：大量实际场景**无动态轨迹、无比对样本、无专业采集设备**；现有鉴定技术大多依赖手写板动态数据；
3. **现有产品缺位**：市面上笔迹鉴别产品大多用于区分人与人手写笔迹，**缺少专门针对AI伪造签名筛查的成熟产品**。

#### 市场规模
中国身份识别与签署市场，2023年规模1523亿元，预计2027年达到3686亿元；
仅北京地区中小企业相关核心场景市场规模约44亿元。

#### 用户调研发现
项目做过100份有效问卷调研：
- 超过半数用户对电子签名安全性心存顾虑；
- 80%以上用户担心AI伪造带来财产泄露、法律纠纷风险；
- 个人用户希望低成本甚至免费的快速核验工具。
""")

    # 子tab3：完整核心技术体系（重点，中期答辩核心）
    with tab_tech:
        box3 = st.container(border=True)
        with box3:
            st.subheader("项目完整核心技术体系三大部分")
            tech_tab1,tech_tab2,tech_tab3,tech_tab4 = st.tabs(["1.双源笔迹数据库构建","2.K3M骨架提取算法","3.四大量化特征详解","4.多特征融合加权判别模型"])
            with tech_tab1:
                st.markdown("""
### 双源笔迹数据库构建技术
1. 使用Wacom CTL‑672电磁手写板，采集18‑28岁本科以上人群真实手写签名，同步保存图像与原始动态轨迹；
2. 使用**SDT AI造字模型**，基于真人笔迹风格，批量生成高仿真AI伪造签名；
3. 筛选国内高频姓名共153个对象，构建真人‑AI成对的双源笔迹数据库，一共数万条签名样本；
4. 数据库预留扩容接口，可以持续扩充不同年龄、不同书写风格样本，用于后续模型迭代优化。
""")
            with tech_tab2:
                st.markdown("""
### K3M七阶段迭代骨架提取算法（项目自研预处理核心）
传统骨架细化算法存在三大缺陷：骨架偏置、过度侵蚀笔画、拓扑结构断裂。

K3M算法实现方案：
1. 七阶段迭代像素细化；
2. 8邻域查找表规则判断需要删除的笔画像素；
3. 自适应小连通域过滤，去除噪点；
4. 断点修复逻辑，还原断裂笔画拓扑；

**骨架作用：剥离笔迹颜色、灰度干扰，只保留笔画中心线，为后续计算笔画宽度、轨迹特征提供基础。**
> 本系统检测模块输出的【K3M笔迹骨架】图片，就是申报书附件原版算法直接输出结果。
""")
            with tech_tab3:
                st.markdown("""
### 四大核心量化特征（全部中间观测参数均对外输出）
#### ①笔画粗细变异系数（权重20%）
基于骨架到笔迹边缘距离反推笔画宽度，输出：变异系数、平均笔画宽度、笔画宽度标准差；
AI伪造签名笔画生成过于规整，变异系数普遍偏低。

#### ②贝塞尔曲线拟合残差（权重25%）
对签名笔画轮廓进行8控制点贝塞尔样条拟合，输出：平均残差、最大残差、有效笔画长度；
AI伪造签名本身就是贝塞尔曲线生成，拟合残差会显著小于真人书写。

#### ③书写速度等效波动熵（权重35%，权重最高）
**不需要动态手写板数据！** 使用道格拉斯‑普克算法提取骨架上关键特征点；
输出指标：等效速度变异系数、特征点密度、等效速度均值、等效速度标准差、波动熵、有效笔画长度；
真人书写受手部生理抖动，运笔节奏波动更大，熵数值更低；AI生成轨迹平滑，熵更大。

#### ④异常部件发生率（权重20%）
识别笔迹内部飞白、笔画断点、AI生成带来的局部笔画畸变；本版本代码预留接口，中期暂未完全实现。
""")
            with tech_tab4:
                st.latex(r"S = 0.20 \cdot X_1 + 0.25 \cdot X_2 + 0.35 \cdot X_3 + 0.20 \cdot X_4")
                st.markdown(r"""
### 多特征融合加权判别模型
$X_1$：笔画粗细变异系数
$X_2$：贝塞尔曲线拟合残差（归一化）
$X_3$：书写速度等效波动熵（归一化）
$X_4$：异常部件发生率（预留）

#### 判别阈值
- $S < 0.42$ → 高度疑似AI伪造电子签名
- $S > 0.62$ → 高度疑似真人手写电子签名
- $0.42 \le S \le 0.62$ → 过渡灰色区间，强制人工复核

> 算法全部轻量化实现，可以部署网页、小程序，不需要GPU，普通CPU即可完成计算；网页端所有计算逻辑与申报书附件独立Python源码一一对应。
""")

    # 子tab4：实验数据可视化
    with tab_exp_chart:
        box4 = st.container(border=True)
        with box4:
            st.subheader("📊项目实验统计可视化图表（来自双源笔迹数据库）")
            exp_labels = ["笔画粗细变异系数","贝塞尔平均残差(px)","书写等效波动熵"]
            ai_group_data = [0.5302,25.73,-2.9651]
            human_group_data = [0.5627,70.59,-3.8227]
            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(x=exp_labels,y=ai_group_data,name="AI伪造签名（实验均值）",marker_color="#2b5797"))
            fig_bar.add_trace(go.Bar(x=exp_labels,y=human_group_data,name="真人手写签名（实验均值）",marker_color="#c82423"))
            fig_bar.update_layout(
                title="AI伪造签名组 VS 真人签名组指标均值对比",
                height=440,
                template="plotly_white"
            )
            st.plotly_chart(fig_bar,use_container_width=True)
            st.caption("实验统计样本说明：AI伪造样本n=153，真人手写样本n=20；来源于项目自建双源笔迹数据库")
            st.markdown("""
> 📚课堂教学使用提示：
> 1. 可以让学生切换回到【签名检测模块】上传样本；
> 2. 将学生样本算出全套指标和本实验柱状图进行对比；
> 3. 完整观测中间参数，直观感受AI签名与真人签名在各项特征上的统计差异。
""")

    # 子tab5：商业模式介绍
    with tab_business:
        box5 = st.container(border=True)
        with box5:
            st.subheader("💼项目商业模式")
            st.markdown("""
### 四大目标客群以及对应服务模式
1. **中小企业**：网页/小程序批量筛查，基础免费+订阅付费，解决合同签名风控；
2. **司法鉴定机构**：技术授权，作为案件前期初筛辅助工具，**不替代正式司法鉴定意见书**；
3. **高等院校**：实训模型授权，提供课堂演示、学生实操工具，弥补高校缺少AI伪造笔迹实训载体的现状；
4. **个人用户**：网页免费基础快筛，付费获取完整PDF分析报告。

### 盈利来源
中小企业订阅服务费、司法鉴定机构技术授权费、高校实训模型授权费、个人增值服务费。

### 竞争定位
不和传统司法鉴定机构、大型安全厂商正面竞争，抢占**基层前端快速筛查**市场空白，定位是前端辅助工具。
""")

    # 子tab6：项目落地成果与未来展望
    with tab_achievement:
        box6 = st.container(border=True)
        with box6:
            st.subheader("🏆项目现阶段落地成果")
            st.divider()
            st.subheader("📜项目主体资质证明")
            st.markdown("""
为进一步明确项目主体、规范科研教学演示使用边界，系统展示项目所属主体营业执照信息。
本图片仅作为**项目主体与教学演示证明材料**使用，不代表司法鉴定资质，不替代正式文件。
""")

            try:
                license_img = PILImage.open("license.png")
                st.image(license_img, caption="营业执照（项目主体证明）", width=460)
            except Exception:
                st.warning("未检测到本地营业执照图片 license.png，已跳过展示。")
                st.markdown("""
1. ✅已经获得北京某科技有限公司5万元项目投资；
2. ✅和北京某高校侦查相关学院达成课堂实训模型试用意向；
3. ✅和北京某信息咨询有限公司达成初步合作试用意向；
4. ✅项目公众号「当AI开始练字」已经发布7篇科普推文，总浏览1403人次，做公众科普引流；
5. ✅完成申报书附件全套算法工程化，开发本网页教学演示系统，完整输出全部实验中间参数，可直接用于课堂演示、中期答辩；

### 未来发展规划
1. 依托学校科技园推进公司注册；
2. 迭代完善异常部件发生率特征模块，完善模型；
3. 拓展更多高校、中小企业试点；
4. 持续运营公众号科普，扩大项目社会影响力；
5. 完成微信小程序版本开发，拓展产品载体。
""")

# 页脚
st.markdown("""
<div class="footer-note">
©2026 北京市大学生创新训练项目｜真迹云鉴｜V8.1<br>
本系统仅供科研教学演示，不具备司法鉴定法律效力
</div>
""",unsafe_allow_html=True)


# =====================程序入口，自动区分CLI控制台 / Streamlit网页运行模式=====================
if __name__ == "__main__":
    import sys
    cli_args = sys.argv
    is_streamlit_mode = any("streamlit" in arg for arg in cli_args)
    if not is_streamlit_mode:
        run_cli_console()
