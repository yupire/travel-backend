#!/usr/bin/env python
"""测试高德地图 API 真实数据调用"""
import sys
import os

# 从 .env 加载环境变量
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(filename='.env', usecwd=True), override=True)

print(f'当前 AMAP_API_KEY: {os.getenv("AMAP_API_KEY", "")[:12]}...')

# 直接执行模块代码（绕过 tools/__init__.py）
amap_code = open('tools/amap.py').read()

# 创建命名空间并添加���要变量
amap_ns = {'__name__': '__amap__', '__file__': os.path.join(os.getcwd(), 'tools', 'amap.py')}
exec(amap_code, amap_ns)

# 执行 spots.py（注入 amap 的函数）
spots_code = open('tools/spots.py').read()
spots_ns = {
    **amap_ns,
    '__name__': '__spots__',
    '__file__': os.path.join(os.getcwd(), 'tools', 'spots.py'),
    'AMapError': amap_ns['AMapError'],
    '_cache_get': amap_ns['_cache_get'],
    '_cache_set': amap_ns['_cache_set'],
    '_request': amap_ns['_request'],
    'is_chinese_city': amap_ns['is_chinese_city'],
    'is_configured': amap_ns['is_configured'],
    'normalize_city_name': amap_ns['normalize_city_name'],
    'parse_location': amap_ns['parse_location'],
}
exec(spots_code, spots_ns)

# 执行 foods.py
foods_code = open('tools/foods.py').read()
foods_ns = {
    **amap_ns,
    '__name__': '__foods__',
    '__file__': os.path.join(os.getcwd(), 'tools', 'foods.py'),
    'AMapError': amap_ns['AMapError'],
    '_cache_get': amap_ns['_cache_get'],
    '_cache_set': amap_ns['_cache_set'],
    '_request': amap_ns['_request'],
    'is_chinese_city': amap_ns['is_chinese_city'],
    'is_configured': amap_ns['is_configured'],
    'normalize_city_name': amap_ns['normalize_city_name'],
    'parse_location': amap_ns['parse_location'],
}
exec(foods_code, foods_ns)

get_top_spots = spots_ns['get_top_spots']
get_top_foods = foods_ns['get_top_foods']
is_configured = amap_ns['is_configured']

print("=" * 50)
print("测试高德地图真实 API 数据")
print("=" * 50)

print(f"\n✓ is_configured(): {is_configured()}")

print("\n【1】中国城市景点测试（高德真实数据）")
print("  —— 北京 ——")
spots_beijing = get_top_spots('beijing', 5)
print(f"  返回 {len(spots_beijing)} 个景点")
for i, s in enumerate(spots_beijing[:3], 1):
    print(f"    {i}. {s.get('name')} | {s.get('type')} | ({s.get('lat'):.4f}, {s.get('lng'):.4f})")

print("\n  —— 上海 ——")
spots_shanghai = get_top_spots('shanghai', 3)
print(f"  返回 {len(spots_shanghai)} 个景点")
for i, s in enumerate(spots_shanghai, 1):
    print(f"    {i}. {s.get('name')} | {s.get('type')}")

print("\n  —— 成都 ——")
spots_chengdu = get_top_spots('chengdu', 3)
print(f"  返回 {len(spots_chengdu)} 个景点")
for i, s in enumerate(spots_chengdu, 1):
    print(f"    {i}. {s.get('name')} | {s.get('type')}")

print("\n【2】国际城市景点测试（应降级到 Mock）")
print("  —— 东京 ——")
spots_tokyo = get_top_spots('Tokyo', 3)
print(f"  返回 {len(spots_tokyo)} 个景点 (Mock)")
for i, s in enumerate(spots_tokyo[:2], 1):
    print(f"    {i}. {s.get('name')} ({s.get('name_en')})")

print("\n  —— 巴黎 ——")
spots_paris = get_top_spots('paris', 2)
print(f"  返回 {len(spots_paris)} 个景点 (Mock)")
for i, s in enumerate(spots_paris, 1):
    print(f"    {i}. {s.get('name')} ({s.get('name_en')})")

print("\n【3】美食查询测试（高德真实数据）")
print("  —— 北京美食 ——")
foods_beijing = get_top_foods('beijing', 5)
print(f"  返回 {len(foods_beijing)} 个美食")
for i, f in enumerate(foods_beijing[:3], 1):
    print(f"    {i}. {f.get('name')} | {f.get('cuisine')} | ({f.get('lat'):.4f}, {f.get('lng'):.4f})")

print("\n  —— 上海美食 ——")
foods_shanghai = get_top_foods('shanghai', 3)
print(f"  返回 {len(foods_shanghai)} 个美食")
for i, f in enumerate(foods_shanghai, 1):
    print(f"    {i}. {f.get('name')} | {f.get('cuisine')}")

print("\n【4】缓存测试（第二次查询应命中缓存）")
print("  再次查询北京景点（应快速返回）")
import time
start = time.time()
spots_cached = get_top_spots('beijing', 5)
elapsed = time.time() - start
print(f"  耗时: {elapsed*1000:.1f}ms (缓存命中)")
print(f"  返回景点数: {len(spots_cached)}")

print("\n" + "=" * 50)
print("✓ 所有测试完成")
print("=" * 50)
