#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""用户认证 + 识别记录。

职责
----
* 注册 / 登录 / 登出，密码只存哈希（werkzeug 的 generate_password_hash）。
* 识别记录写入：各业务模块预测成功后调用 ``save_upload_record``。
* 记录查询与分页：``query_user_uploads`` / ``get_user_uploads`` 供“我的记录”页使用。
"""

import sqlite3
import os
from pathlib import Path
from flask import request, jsonify, session, redirect, render_template
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / 'database' / 'plant_system.db'


def get_db_connection():
    """获取数据库连接"""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_user_db():
    """初始化用户数据库"""
    conn = get_db_connection()
    cursor = conn.cursor()

    # 时间统一存 UTC（CURRENT_TIMESTAMP），展示时再转成本地时区（如东八区）
    # 创建用户表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            email TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # 创建预测记录表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_uploads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_name TEXT NOT NULL,
            image_name TEXT NOT NULL,
            image_path TEXT NOT NULL,
            disease_class TEXT NOT NULL,
            model_type TEXT,
            confidence REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # 创建索引
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_user_name ON user_uploads(user_name)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_created_at ON user_uploads(created_at)')

    conn.commit()
    conn.close()


def user_endpoints(app):
    """注册用户相关的路由"""

    @app.route('/register', methods=['GET', 'POST'])
    def register():
        """用户注册"""
        if request.method == 'GET':
            return render_template('register.html')

        data = request.get_json()
        username = data.get('username', '').strip()
        password = data.get('password', '').strip()
        email = data.get('email', '').strip()

        if not username or not password:
            return jsonify({'success': False, 'error': '用户名和密码不能为空'}), 400

        if len(password) < 6:
            return jsonify({'success': False, 'error': '密码长度至少6位'}), 400

        try:
            conn = get_db_connection()
            cursor = conn.cursor()

            # 检查用户是否已存在
            cursor.execute('SELECT id FROM users WHERE username = ?', (username,))
            if cursor.fetchone():
                conn.close()
                return jsonify({'success': False, 'error': '用户名已存在'}), 400

            # 插入新用户
            password_hash = generate_password_hash(password)
            cursor.execute(
                'INSERT INTO users (username, password_hash, email) VALUES (?, ?, ?)',
                (username, password_hash, email)
            )

            conn.commit()
            conn.close()

            return jsonify({'success': True, 'message': '注册成功！'})

        except Exception as e:
            print(f"注册错误: {e}")
            return jsonify({'success': False, 'error': '注册失败，请稍后重试'}), 500

    @app.route('/login', methods=['POST'])
    def login():
        """用户登录"""
        data = request.get_json()
        username = data.get('username', '').strip()
        password = data.get('password', '').strip()

        if not username or not password:
            return jsonify({'success': False, 'error': '用户名和密码不能为空'}), 400

        try:
            conn = get_db_connection()
            cursor = conn.cursor()

            cursor.execute('SELECT * FROM users WHERE username = ?', (username,))
            user = cursor.fetchone()
            conn.close()

            if user and check_password_hash(user['password_hash'], password):
                session['user_id'] = user['id']
                session['login_name'] = user['username']
                return jsonify({'success': True, 'message': '登录成功'})
            else:
                return jsonify({'success': False, 'error': '用户名或密码错误'}), 401

        except Exception as e:
            print(f"登录错误: {e}")
            return jsonify({'success': False, 'error': '登录失败，请稍后重试'}), 500

    @app.route('/logout')
    def logout():
        """用户登出"""
        session.clear()
        return redirect('/')

    @app.route('/api/current_user', methods=['GET'])
    def get_current_user():
        """获取当前登录用户信息"""
        if 'login_name' in session:
            return jsonify({
                'logged_in': True,
                'username': session['login_name'],
                'user_id': session.get('user_id')
            })
        return jsonify({'logged_in': False})

    @app.route('/check_login', methods=['GET'])
    def check_login():
        """检查用户登录状态（前端兼容性路由）"""
        if 'login_name' in session:
            return jsonify({'login': True, 'username': session['login_name']})
        return jsonify({'login': False})


def save_upload_record(user_name, image_name, image_path, disease_class, model_type='', confidence=0.0):
    """保存上传记录到数据库"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute(
            '''INSERT INTO user_uploads
               (user_name, image_name, image_path, disease_class, model_type, confidence)
               VALUES (?, ?, ?, ?, ?, ?)''',
            (user_name, image_name, image_path, disease_class, model_type, confidence)
        )

        conn.commit()
        conn.close()
        return True

    except Exception as e:
        print(f"保存上传记录错误: {e}")
        return False


def get_user_uploads(user_name, limit=100):
    """获取用户的上传记录"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute(
            '''SELECT * FROM user_uploads
               WHERE user_name = ?
               ORDER BY created_at DESC
               LIMIT ?''',
            (user_name, limit)
        )

        uploads = cursor.fetchall()
        conn.close()

        return [_normalise_record(row) for row in uploads]

    except Exception as e:
        print(f"获取上传记录错误: {e}")
        return []


def _normalise_record(row):
    """把 sqlite3.Row 转为 dict，并统一图片路径。"""
    item = dict(row)
    path = str(item.get('image_path') or '').replace('\\', '/')
    if path and not path.startswith('/'):
        path = '/' + path
    item['image_path'] = path
    item['upload_time'] = item.get('created_at')
    return item


def query_user_uploads(user_name, page=1, per_page=9, model_type=None, disease_class=None):
    """分页 + 筛选查询用户记录，返回 (rows, total)。"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        where = ['user_name = ?']
        params = [user_name]
        if model_type:
            where.append('model_type = ?')
            params.append(model_type)
        if disease_class:
            where.append('disease_class LIKE ?')
            params.append(f'%{disease_class}%')

        where_sql = ' AND '.join(where)
        cursor.execute(f'SELECT COUNT(*) AS total FROM user_uploads WHERE {where_sql}', params)
        total = int(cursor.fetchone()['total'] or 0)

        page = max(1, int(page))
        per_page = max(1, int(per_page))
        offset = (page - 1) * per_page
        cursor.execute(
            f'''SELECT * FROM user_uploads
                WHERE {where_sql}
                ORDER BY created_at DESC, id DESC
                LIMIT ? OFFSET ?''',
            params + [per_page, offset]
        )
        rows = cursor.fetchall()
        conn.close()
        return [_normalise_record(row) for row in rows], total

    except Exception as e:
        print(f"查询上传记录错误: {e}")
        return [], 0


# 初始化数据库
init_user_db()

# 创建测试用户
try:
    conn = get_db_connection()
    cursor = conn.cursor()

    # 检查是否已有用户
    cursor.execute('SELECT COUNT(*) as count FROM users')
    count = cursor.fetchone()['count']

    if count == 0:
        # 添加测试用户
        cursor.execute(
            'INSERT INTO users (username, password_hash, email) VALUES (?, ?, ?)',
            ('admin', generate_password_hash('admin123'), 'admin@example.com')
        )
        cursor.execute(
            'INSERT INTO users (username, password_hash, email) VALUES (?, ?, ?)',
            ('test', generate_password_hash('test123'), 'test@example.com')
        )
        conn.commit()
        print("[OK] 测试用户创建完成: admin/admin123, test/test123")

    conn.close()
except Exception as e:
    print(f"初始化测试用户错误: {e}")
