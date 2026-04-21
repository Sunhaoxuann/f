import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import Polynomial
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

# 使用 FontProperties 设置字体
import os
import matplotlib
from matplotlib.font_manager import FontProperties
font_path = os.path.join(os.path.dirname(__file__), 'YaHei.ttf')
my_font = FontProperties(fname=font_path)
matplotlib.rcParams['font.sans-serif'] = [my_font.get_name()]
matplotlib.rcParams['axes.unicode_minus'] = False

# 读取所有工作表的数据
file_path = 'total.xlsx'
sheets = pd.read_excel(file_path, sheet_name=None, header=None)

def strict_clean_data(data):
    # 将第一行作为列名
    data.columns = data.iloc[0]
    data = data[1:]
    
    # 转换数据类型
    data = data.apply(pd.to_numeric, errors='coerce')
    
    # 计算总加速度
    data['Total_Acceleration'] = np.sqrt(data['Accelerometer_x']**2 + 
                                        data['Accelerometer_y']**2 + 
                                        data['Accelerometer_z']**2)
    
    # 严格识别自由落体阶段（加速度接近重力加速度且变化平缓）
    gravity = 9.8
    g = 0.5  # 定义一个容忍范围
    free_fall_data = data[(data['Total_Acceleration'] > gravity - g) & 
                            (data['Total_Acceleration'] < gravity + g) &
                            (data['Accelerometer_z'] < 0)]
    
    return free_fall_data

# 对所有工作表进行严格的数据清理
strict_cleaned_sheets = {sheet_name: strict_clean_data(data) for sheet_name, data in sheets.items()}

# 合并所有工作表中的有效数据点
combined_data = pd.concat(strict_cleaned_sheets.values(), ignore_index=True)

def calculate_velocity_and_drag_strict(data):
    gravity = 9.8  # 定义重力加速度
    data['Velocity'] = 0.0
    for i in range(1, len(data)):
        time_interval = data['Accelerometer_time'].iloc[i] - data['Accelerometer_time'].iloc[i-1]
        avg_acceleration = (data['Total_Acceleration'].iloc[i] + 
                            data['Total_Acceleration'].iloc[i-1]) / 2
        # 修正这里：使用正确的loc赋值语法
        data.loc[data.index[i], 'Velocity'] = data['Velocity'].iloc[i-1] + avg_acceleration * time_interval
    
    data['Air_Drag'] = data['Total_Acceleration'] - gravity
    
    return data

combined_data = calculate_velocity_and_drag_strict(combined_data)

# 去除NaN值
combined_data = combined_data.dropna(subset=['Velocity', 'Air_Drag'])

# 准备数据
x_data = combined_data['Velocity'].values
y_data = combined_data['Air_Drag'].values

# 定义拟合函数
def linear_func(x, a, b):
    return a * x + b

def quadratic_func(x, a, b, c):
    return a * x**2 + b * x + c

def power_law_func(x, a, b):
    return a * x**b

def exp_func(x, a, b, c):
    return a * np.exp(b * x) + c

def log_func(x, a, b):
    return a * np.log(x + 1e-10) + b  # 避免log(0)错误

def physics_func(x, k1, k2):
    return k1 * x + k2 * x**2

def rational_func(x, a, b, c):
    """有理函数模型: y = (a + b*x) / (1 + c*x)"""
    return (a + b * x) / (1 + c * x)

def hyperbolic_func(x, a, b):
    """双曲线模型: y = (a * x) / (b + x)"""
    return (a * x) / (b + x)

# 存储拟合结果和R²值
fit_results = {
    '线性拟合': {'func': linear_func, 'params': None, 'r2': None, 'color': 'blue'},
    '二次拟合': {'func': quadratic_func, 'params': None, 'r2': None, 'color': 'green'},
    '幂律拟合': {'func': power_law_func, 'params': None, 'r2': None, 'color': 'purple'},
    '指数拟合': {'func': exp_func, 'params': None, 'r2': None, 'color': 'orange'},
    '对数拟合': {'func': log_func, 'params': None, 'r2': None, 'color': 'brown'},
    '物理模型': {'func': physics_func, 'params': None, 'r2': None, 'color': 'cyan'},
    '有理函数': {'func': rational_func, 'params': None, 'r2': None, 'color': 'magenta'},
    '双曲线': {'func': hyperbolic_func, 'params': None, 'r2': None, 'color': 'olive'}
}

# 进行各种拟合
for name, model in fit_results.items():
    try:
        # 为不同模型提供不同的初始参数
        if name == '指数拟合':
            p0 = [0.1, 0.1, 0.1]  # 指数拟合初始参数
        elif name == '幂律拟合':
            p0 = [0.1, 2]  # 幂律拟合初始参数
        elif name == '物理模型':
            p0 = [0.1, 0.1]  # 物理模型初始参数
        elif name == '有理函数':
            p0 = [0.1, 0.1, 0.1]  # 有理函数初始参数
        elif name == '双曲线':
            p0 = [1.0, 1.0]  # 双曲线初始参数
        else:
            p0 = None  # 其他模型使用默认初始参数
        
        # 尝试拟合
        if p0:
            params, _ = curve_fit(model['func'], x_data, y_data, p0=p0, maxfev=10000)
        else:
            params, _ = curve_fit(model['func'], x_data, y_data, maxfev=10000)
        
        # 计算拟合值和R²
        y_fit = model['func'](x_data, *params)
        r2 = r2_score(y_data, y_fit)
        
        model['params'] = params
        model['r2'] = r2
    except Exception as e:
        print(f"{name}拟合失败: {str(e)}")
        model['params'] = None
        model['r2'] = np.nan

# 找出最优拟合（R²最高的模型）
best_fit_name = None
best_fit_r2 = -np.inf
best_fit_params = None

for name, result in fit_results.items():
    if result['r2'] is not None and not np.isnan(result['r2']) and result['r2'] > best_fit_r2:
        best_fit_name = name
        best_fit_r2 = result['r2']
        best_fit_params = result['params']

# 打印各种拟合的参数和R²值
print("\n各种拟合模型的参数和拟合优度:")
for name, result in fit_results.items():
    if result['params'] is not None:
        params = result['params']
        equation = ""
        
        if name == '线性拟合':
            equation = f"y = {params[0]:.6f}x + {params[1]:.6f}"
        elif name == '二次拟合':
            equation = f"y = {params[0]:.6f}x² + {params[1]:.6f}x + {params[2]:.6f}"
        elif name == '幂律拟合':
            equation = f"y = {params[0]:.6f}x^{params[1]:.6f}"
        elif name == '指数拟合':
            equation = f"y = {params[0]:.6f}·e^{params[1]:.6f}x + {params[2]:.6f}"
        elif name == '对数拟合':
            equation = f"y = {params[0]:.6f}·ln(x) + {params[1]:.6f}"
        elif name == '物理模型':
            equation = f"y = {params[0]:.6f}v + {params[1]:.6f}v²"
        elif name == '有理函数':
            equation = f"y = ({params[0]:.6f} + {params[1]:.6f}x)/(1 + {params[2]:.6f}x)"
        elif name == '双曲线':
            equation = f"y = ({params[0]:.6f}·x)/({params[1]:.6f} + x)"
        
        r2_str = f"{result['r2']:.6f}"
        print(f"{name}: {equation}, R²={r2_str}")

if best_fit_name:
    print(f"\n最优拟合模型: {best_fit_name}, R²={best_fit_r2:.6f}")
else:
    print("\n没有找到有效的拟合模型")

# 生成用于绘制拟合曲线的数据点
velocity_range = np.linspace(max(0.1, x_data.min()), x_data.max(), 500)  # 避免对数拟合的0值问题

# 修改第一张图表的绘制部分
plt.figure(figsize=(16, 10))
plt.scatter(x_data, y_data, label='原始数据点', color='black', alpha=0.5, s=20)

# 绘制各种拟合曲线
linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':']
for i, (name, model) in enumerate(fit_results.items()):
    if model['params'] is not None:
        func = model['func']
        params = model['params']
        color = model['color']
        ls = linestyles[i % len(linestyles)]
        
        # 只显示模型名称，不显示R²
        plt.plot(velocity_range, func(velocity_range, *params), 
                label=f'{name}',  # 仅显示模型名称
                color=color, linestyle=ls, linewidth=2)

# 设置图表属性
plt.xlabel('速度 (m/s)', fontsize=14)
plt.ylabel('空气阻力 (N)', fontsize=14)
plt.title('不同模型拟合空气阻力与速度的关系', fontsize=16)
plt.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# 修改第二张图表的绘制部分
if best_fit_name:
    plt.figure(figsize=(10, 6))
    plt.scatter(x_data, y_data, label='原始数据点', color='black', alpha=0.5, s=20)

    best_model = fit_results[best_fit_name]
    best_func = best_model['func']
    best_params = best_model['params']
    best_color = best_model['color']
    
    plt.plot(velocity_range, best_func(velocity_range, *best_params), 
            label=f'{best_fit_name}',  # 仅显示模型名称
            color=best_color, linewidth=3)
    
    # 生成方程表达式（保持不变）
    if best_fit_name == '线性拟合':
        equation = f"y = {best_params[0]:.4f}x + {best_params[1]:.4f}"
    elif best_fit_name == '二次拟合':
        equation = f"y = {best_params[0]:.4f}x² + {best_params[1]:.4f}x + {best_params[2]:.4f}"
    elif best_fit_name == '幂律拟合':
        equation = f"y = {best_params[0]:.4f}x^{best_params[1]:.4f}"
    elif best_fit_name == '指数拟合':
        equation = f"y = {best_params[0]:.4f}·e^{best_params[1]:.4f}x + {best_params[2]:.4f}"
    elif best_fit_name == '对数拟合':
        equation = f"y = {best_params[0]:.4f}·ln(x) + {best_params[1]:.4f}"
    elif best_fit_name == '物理模型':
        equation = f"y = {best_params[0]:.4f}v + {best_params[1]:.4f}v²"
    elif best_fit_name == '有理函数':
        equation = f"y = ({best_params[0]:.4f} + {best_params[1]:.4f}x)/(1 + {best_params[2]:.4f}x)"
    elif best_fit_name == '双曲线':
        equation = f"y = ({best_params[0]:.4f}·x)/({best_params[1]:.4f} + x)"
    
    plt.annotate(equation, xy=(0.05, 0.95), xycoords='axes fraction', 
                fontsize=12, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

    plt.xlabel('速度 (m/s)', fontsize=14)
    plt.ylabel('空气阻力 (N)', fontsize=14)
    plt.title(f'最优拟合模型: {best_fit_name}', fontsize=16)
    plt.legend(fontsize=10, loc='best')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()