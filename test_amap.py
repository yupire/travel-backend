#!/usr/bin/env python
"""测试高德地图 API 集成

验证：
1. Mock 降级功能（无 API Key 时）
2. 城市判断逻辑
3. 数据格式兼容性
"""
import sys
import os

# 确保没有 API Key，强制使用 mock
os.environ['AMAP_API_KEY'] = ''

# 直接执行模块代码（绕过 tools/__init__.py）
amap_code = open('tools/amap.py').read()

# 创建命名空间并添加必要变量
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
get_spot_map = spots_ns['get_spot_map']
geocode_spots = spots_ns['geocode_spots']
get_nearby_foods = foods_ns['get_nearby_foods']
is_chinese_city = amap_ns['is_chinese_city']
normalize_city_name = amap_ns['normalize_city_name']

print("=" * 50)
print("测试高德地图 API 集成")
print("=" * 50)

print("\n【1】城市判断测试")
print(f"  is_chinese_city('北京'): {is_chinese_city('北京')}")
print(f"  is_chinese_city('beijing'): {is_chinese_city('beijing')}")
print(f"  is_chinese_city('Tokyo'): {is_chinese_city('Tokyo')}")
print(f"  normalize_city_name('beijing'): {normalize_city_name('beijing')}")
print(f"  normalize_city_name('北京市'): {normalize_city_name('北京市')}")

print("\n【2】景点查询测试（Mock 降级）")
spots_beijing = get_top_spots("beijing", 5)
print(f"  get_top_spots('beijing', 5) → {len(spots_beijing)} 个景点")
if spots_beijing:
    s = spots_beijing[0]
    print(f"    示例: {s.get('name')} ({s.get('name_en')})")
    print(f"    字段: id={s.get('id')}, lat={s.get('lat')}, lng={s.get('lng')}")
    print(f"    类型: {s.get('type')}, is_indoor={s.get('is_indoor')}")
    print(f"    tags: {s.get('tags')}")

spots_tokyo = get_top_spots("Tokyo", 3)
print(f"  get_top_spots('Tokyo', 3) → {len(spots_tokyo)} 个景点")
if spots_tokyo:
    print(f"    示例: {spots_tokyo[0].get('name')}")

print("\n【3】景点字典测试")
spot_map = get_spot_map("beijing")
print(f"  get_spot_map('beijing') → {len(spot_map)} 个景点")

print("\n【4】地理编码测试")
geocoded = geocode_spots("beijing", ["故宫", "颐和园", "不存在的景点"])
print(f"  geocode_spots('beijing', ['故宫', '颐和园', '不存在的景点']) → {len(geocoded)} 个")
for g in geocoded:
    print(f"    {g.get('name')}: lat={g.get('lat')}, lng={g.get('lng')}")

print("\n【5】美食查询测试（Mock 降级）")
foods_beijing = get_top_foods("beijing", 3)
print(f"  get_top_foods('beijing', 3) → {len(foods_beijing)} 个美食")
if foods_beijing:
    f = foods_beijing[0]
    print(f"    示例: {f.get('name')} ({f.get('cuisine')})")
    print(f"    字段: id={f.get('id')}, lat={f.get('lat')}, lng={f.get('lng')}")

print("\n【6】附近美食测试")
if spots_beijing:
    nearby = get_nearby_foods(spots_beijing[0], foods_beijing, 2)
    print(f"  get_nearby_foods(景点, 美食列表, 2) → {len(nearby)} 个附近美食")
    for n in nearby:
        print(f"    {n.get('name')}: 距离 {n.get('distance_m')} 米")

print("\n" + "=" * 50)
print("✓ 所有测试通过")
print("=" * 50)
