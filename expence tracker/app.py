import os
import sqlite3
import pandas as pd
import numpy as np
import smtplib
from flask import Flask, render_template, request, redirect, session, url_for
from flask_mail import Mail, Message
import matplotlib.pyplot as plt
from datetime import datetime
import pytz
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
from reportlab.lib.styles import ParagraphStyle
from sklearn.linear_model import LinearRegression

# Initialize Flask app
app = Flask(__name__)
app.secret_key = "Akash@_203"

# Email Configuration
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'your_email@gmail.com'  # Replace with your email
app.config['MAIL_PASSWORD'] = 'your_email_password'  # Replace with your email password
mail = Mail(app)

# File Upload Configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'pdf', 'csv'}

ALLOWED_EXTENSIONS = {'pdf', 'csv', 'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Database Initialization
def init_db():
    conn = sqlite3.connect('expense_tracker.db')
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY,
        email TEXT UNIQUE,
        password TEXT,
        name TEXT,
        phone TEXT,
        upi_id TEXT,
        bank_acc TEXT,
        ifsc_code TEXT,
        profile_pic TEXT
    )''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS transactions (
        id INTEGER PRIMARY KEY,
        user_id INTEGER,
        type TEXT,
        category TEXT,
        amount REAL,
        date TEXT,
        FOREIGN KEY (user_id) REFERENCES users (id)
    )''')
    conn.commit()
    conn.close()

init_db()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_transactions(file_path):
    transactions = []
    
    if file_path.endswith(".pdf"):
        import pdfplumber
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    lines = text.split("\n")
                    for line in lines:
                        parts = line.split()
                        if len(parts) >= 3:
                            transactions.append({
                                "date": parts[0],  
                                "description": " ".join(parts[1:-1]),  
                                "amount": float(parts[-1]),  
                                "category": "Uncategorized"
                            })
    
    elif file_path.endswith(".csv"):
        import csv
        with open(file_path, newline='', encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                transactions.append({
                    "date": row.get("Date", ""),
                    "description": row.get("Description", ""),
                    "amount": float(row.get("Amount", 0)),
                    "category": row.get("Category", "Uncategorized")
                })
    
    return transactions

@app.route('/upload_transactions', methods=['POST'])
def upload_transactions():
    if 'user_id' not in session:
        return redirect('/login')

    if 'file' not in request.files:
        return "No file selected", 400

    file = request.files['file']
    
    if file and allowed_file(file.filename):
        from werkzeug.utils import secure_filename
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Extract transactions
        transactions = extract_transactions(file_path)

        # Store transactions in database
        user_id = session['user_id']
        conn = sqlite3.connect('expense_tracker.db')
        cursor = conn.cursor()
        for tx in transactions:
            cursor.execute('''
                INSERT INTO transactions (user_id, type, category, amount, date, description)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (user_id, "expense", tx["category"], tx["amount"], tx["date"], tx["description"]))
        conn.commit()
        conn.close()

        return redirect('/dashboard')

    return "Invalid file format", 400

# Home Route
@app.route('/')
def home():
    return render_template('index.html')

# Registration Route
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form.get('email')
        password = generate_password_hash(request.form.get('password'))  # Hashing password
        name = request.form.get('name')

        conn = sqlite3.connect('expense_tracker.db')
        cursor = conn.cursor()
        try:
            cursor.execute('INSERT INTO users (email, password, name) VALUES (?, ?, ?)', (email, password, name))
            conn.commit()
            return redirect('/login')
        except sqlite3.IntegrityError:
            return "Email already exists."
        finally:
            conn.close()
    return render_template('register.html')

# Login Route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        conn = sqlite3.connect('expense_tracker.db')
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM users WHERE email = ?', (email,))
        user = cursor.fetchone()
        conn.close()

        if user and check_password_hash(user[2], password):  # Verifying password hash
            session['user_id'] = user[0]
            session['profile_pic'] = user[8] if user[8] else None
            return redirect('/dashboard')
        else:
            return "Invalid email or password."

    return render_template('login.html')

# Logout Route
@app.route('/logout')
def logout():
    session.clear()
    return redirect('/login')

# Dashboard Route
@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect('/login')

    user_id = session['user_id']
    conn = sqlite3.connect('expense_tracker.db')
    cursor = conn.cursor()
    cursor.execute('SELECT SUM(amount) FROM transactions WHERE user_id = ? AND type = "income"', (user_id,))
    income = cursor.fetchone()[0] or 0

    cursor.execute('SELECT SUM(amount) FROM transactions WHERE user_id = ? AND type = "expense"', (user_id,))
    expense = cursor.fetchone()[0] or 0

    cursor.execute('SELECT SUM(amount) FROM transactions WHERE user_id = ? AND type = "saving"', (user_id,))
    savings = cursor.fetchone()[0] or 0

    balance = income - (expense + savings)
    outstanding = 0 if balance >= 0 else abs(balance)
    balance = max(0, balance)

    # Generate bar chart
    categories = ['Income', 'Expense', 'Savings']
    amounts = [income, expense, savings]
    plt.figure(figsize=(10, 6))
    bars = plt.bar(categories, amounts, color=['#4CAF50', '#FF5722', '#2196F3'])
    plt.title('Income vs Expense vs Savings', fontsize=18, fontweight='bold', color='#333')
    plt.xlabel('Categories', fontsize=14)
    plt.ylabel('Amount (₹)', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 100,
                 f'₹{int(yval)}', ha='center', va='bottom', fontsize=12,
                 fontweight='bold', color='black',
                 bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.3'))

    chart_path = 'static/bar_chart.png'
    plt.savefig(chart_path)
    plt.close()

    cursor.execute('SELECT * FROM transactions WHERE user_id = ?', (user_id,))
    transactions = cursor.fetchall()

    cursor.execute('SELECT email, name, phone, upi_id, bank_acc, ifsc_code, profile_pic FROM users WHERE id = ?', (user_id,))
    user_details = cursor.fetchone()

    conn.close()
    return render_template('dashboard.html', income=income, expense=expense, savings=savings, balance=balance, outstanding=outstanding, bar_chart=chart_path, transactions=transactions, user_details=user_details)

# Add Transaction Route
@app.route('/add_transaction', methods=['GET', 'POST'])
def add_transaction():
    if 'user_id' not in session:
        return redirect('/login')

    if request.method == 'POST':
        transaction_type = request.form['type']
        category = request.form['category']
        amount = request.form['amount']

        user_id = session['user_id']
        ist = pytz.timezone('Asia/Kolkata')
        date = datetime.now(ist).strftime('%Y-%m-%d %H:%M:%S')
        conn = sqlite3.connect('expense_tracker.db')
        cursor = conn.cursor()
        cursor.execute('INSERT INTO transactions (user_id, type, category, amount, date) VALUES (?, ?, ?, ?, ?)',
                       (user_id, transaction_type, category, float(amount), date))
        conn.commit()
        conn.close()

        return redirect('/dashboard')

    return render_template('add_transaction.html')

# Profile Picture Upload
@app.route('/upload_profile_pic', methods=['POST'])
def upload_profile_pic():
    if 'user_id' not in session:
        return redirect('/login')

    if 'profile_pic' not in request.files:
        return redirect('/dashboard')

    file = request.files['profile_pic']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        conn = sqlite3.connect('expense_tracker.db')
        cursor = conn.cursor()
        cursor.execute('UPDATE users SET profile_pic = ? WHERE id = ?', (filename, session['user_id']))
        conn.commit()
        conn.close()

        session['profile_pic'] = filename
        return redirect('/dashboard')

# Accounts Sheet Route
@app.route('/accounts_sheet')
def accounts_sheet():
    if 'user_id' not in session:
        return redirect('/login')

    conn = sqlite3.connect('expense_tracker.db')
    cursor = conn.cursor()
    cursor.execute('SELECT name, phone, upi_id, bank_acc, ifsc_code FROM users WHERE id = ?', (session['user_id'],))
    user_details = cursor.fetchone()

    cursor.execute('SELECT type, category, amount, date FROM transactions WHERE user_id = ? ORDER BY date', (session['user_id'],))
    transactions = cursor.fetchall()

    # Convert transaction dates to 12-hour format
    converted_transactions = []
    for t in transactions:
        try:
            date_str = datetime.strptime(t[3], '%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%d %I:%M %p')
        except (ValueError, TypeError):
            date_str = t[3]  # If conversion fails, keep original value
        converted_transactions.append((t[0], t[1], t[2], date_str))

    conn.close()
    return render_template('accounts_sheet.html', user_details=user_details, transactions=converted_transactions)

# Generate PDF
def generate_pdf(user_details, transactions):
    pdf_path = 'static/accounts_sheet.pdf'
    document = SimpleDocTemplate(pdf_path, pagesize=A4)
    elements = []

    # Title
    elements.append(Paragraph("Accounts Sheet", ParagraphStyle('Title', fontSize=24, spaceAfter=20)))

    # User details
    user_data = [
        ["Name", user_details[0]],
        ["Phone", user_details[1]],
        ["UPI ID", user_details[2]],
        ["Bank Account", user_details[3]],
        ["IFSC Code", user_details[4]]
    ]
    table = Table(user_data, colWidths=[100, 300])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    elements.append(table)

    # Transactions
    elements.append(Paragraph("Transactions", ParagraphStyle('Heading2', fontSize=18, spaceBefore=20, spaceAfter=20)))
    transaction_data = [["Type", "Category", "Amount", "Date"]]
    for transaction in transactions:
        transaction_data.append([transaction[0], transaction[1], transaction[2], transaction[3]])

    table = Table(transaction_data, colWidths=[100, 100, 100, 200])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    elements.append(table)

    document.build(elements)
    return pdf_path

# Download Accounts Sheet Route
@app.route('/download_accounts_sheet')
def download_accounts_sheet():
    if 'user_id' not in session:
        return redirect('/login')

    conn = sqlite3.connect('expense_tracker.db')
    cursor = conn.cursor()
    cursor.execute('SELECT name, phone, upi_id, bank_acc, ifsc_code FROM users WHERE id = ?', (session['user_id'],))
    user_details = cursor.fetchone()

    cursor.execute('SELECT type, category, amount, date FROM transactions WHERE user_id = ? ORDER BY date', (session['user_id'],))
    transactions = cursor.fetchall()

    # Convert transaction dates to 12-hour format
    converted_transactions = []
    for t in transactions:
        try:
            date_str = datetime.strptime(t[3], '%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%d %I:%M %p')
        except (ValueError, TypeError):
            date_str = t[3]  # If conversion fails, keep original value
        converted_transactions.append((t[0], t[1], t[2], date_str))

    conn.close()

    pdf_path = generate_pdf(user_details, converted_transactions)

    return redirect(url_for('static', filename='accounts_sheet.pdf'))

# Print Accounts Sheet Route
@app.route('/print_accounts_sheet')
def print_accounts_sheet():
    if 'user_id' not in session:
        return redirect('/login')

    conn = sqlite3.connect('expense_tracker.db')
    cursor = conn.cursor()
    cursor.execute('SELECT name, phone, upi_id, bank_acc, ifsc_code FROM users WHERE id = ?', (session['user_id'],))
    user_details = cursor.fetchone()

    cursor.execute('SELECT type, category, amount, date FROM transactions WHERE user_id = ? ORDER BY date', (session['user_id'],))
    transactions = cursor.fetchall()

    # Convert transaction dates to 12-hour format
    converted_transactions = []
    for t in transactions:
        try:
            date_str = datetime.strptime(t[3], '%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%d %I:%M %p')
        except (ValueError, TypeError):
            date_str = t[3]  # If conversion fails, keep original value
        converted_transactions.append((t[0], t[1], t[2], date_str))

    conn.close()

    pdf_path = generate_pdf(user_details, converted_transactions)

    return render_template('print_accounts_sheet.html', pdf_path=pdf_path)

# Edit Transaction Route
@app.route('/edit_transaction/<int:transaction_id>', methods=['GET', 'POST'])
def edit_transaction(transaction_id):
    if 'user_id' not in session:
        return redirect('/login')

    conn = sqlite3.connect('expense_tracker.db')
    cursor = conn.cursor()

    if request.method == 'POST':
        transaction_type = request.form['type']
        category = request.form['category']
        amount = request.form['amount']
        date = request.form['date']

        # Convert the date from HTML input format to database format
        try:
            date_obj = datetime.strptime(date, '%Y-%m-%dT%H:%M')
            formatted_date = date_obj.strftime('%Y-%m-%d %H:%M:%S')
        except ValueError:
            formatted_date = date  # Fallback to the original date if conversion fails

        cursor.execute('UPDATE transactions SET type = ?, category = ?, amount = ?, date = ? WHERE id = ? AND user_id = ?',
                       (transaction_type, category, amount, formatted_date, transaction_id, session['user_id']))
        conn.commit()
        conn.close()

        return redirect('/dashboard')

    cursor.execute('SELECT * FROM transactions WHERE id = ? AND user_id = ?', (transaction_id, session['user_id']))
    transaction = cursor.fetchone()

    # Ensure the date is correctly formatted for the datetime-local input field
    if transaction and transaction[5]:  # Assuming date is the 6th column (index 5)
        try:
            transaction_date = datetime.strptime(transaction[5], '%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%dT%H:%M')
        except ValueError:
            transaction_date = transaction[5]  # Fallback to the original date if conversion fails
    else:
        transaction_date = ''

    conn.close()
    return render_template('edit_transaction.html', transaction=transaction, transaction_date=transaction_date)

# Delete Transaction Route
@app.route('/delete_transaction/<int:transaction_id>', methods=['POST'])
def delete_transaction(transaction_id):
    if 'user_id' not in session:
        return redirect('/login')

    conn = sqlite3.connect('expense_tracker.db')
    cursor = conn.cursor()
    cursor.execute('DELETE FROM transactions WHERE id = ? AND user_id = ?', (transaction_id, session['user_id']))
    conn.commit()
    conn.close()

    return redirect('/dashboard')
# Edit Profile Route
@app.route('/edit_profile', methods=['GET', 'POST'])
def edit_profile():
    if 'user_id' not in session:
        return redirect('/login')

    if request.method == 'POST':
        email = request.form.get('email')
        name = request.form.get('name')
        phone = request.form.get('phone')
        upi_id = request.form.get('upi_id')
        bank_acc = request.form.get('bank_acc')
        ifsc_code = request.form.get('ifsc_code')

        conn = sqlite3.connect('expense_tracker.db')
        cursor = conn.cursor()    
        cursor.execute('UPDATE users SET email = ?, name = ?, phone = ?, upi_id = ?, bank_acc = ?, ifsc_code = ? WHERE id = ?',
                       (email, name, phone, upi_id, bank_acc, ifsc_code, session['user_id']))
        conn.commit()
        conn.close()

        return redirect('/profile')  # Redirect to the profile page after updating

    conn = sqlite3.connect('expense_tracker.db')
    cursor = conn.cursor()
    cursor.execute('SELECT email, name, phone, upi_id, bank_acc, ifsc_code FROM users WHERE id = ?', (session['user_id'],))
    user_details = cursor.fetchone()
    conn.close()

    return render_template('edit_profile.html', user_details=user_details)

# Profile Route
@app.route('/profile')
def profile():
    if 'user_id' not in session:
        return redirect('/login')

    user_id = session['user_id']
    conn = sqlite3.connect('expense_tracker.db')
    cursor = conn.cursor()
    cursor.execute('SELECT email, name, phone, upi_id, bank_acc, ifsc_code, profile_pic FROM users WHERE id = ?', (user_id,))
    user_details = cursor.fetchone()
    conn.close()

    return render_template('profile.html', user_details=user_details)
    conn = sqlite3.connect('expense_tracker.db')
    cursor = conn.cursor()
    cursor.execute('SELECT email, name, phone, upi_id, bank_acc, ifsc_code FROM users WHERE id = ?', (session['user_id'],))
    user_details = cursor.fetchone()
    conn.close()

    return render_template('edit_profile.html', user_details=user_details)
    # Utility functions
def get_user_transactions(user_id):
    conn = sqlite3.connect('expense_tracker.db')
    cursor = conn.cursor()
    cursor.execute("SELECT date, amount FROM transactions WHERE user_id = ? AND type = 'expense' ORDER BY date", (user_id,))
    transactions = cursor.fetchall()
    conn.close()
    return transactions

# AI-Based Insights
def analyze_spending_pattern(user_id):
    transactions = get_user_transactions(user_id)
    if not transactions:
        return "No data available for analysis.", None
    
    df = pd.DataFrame(transactions, columns=['date', 'amount'])
    df['date'] = pd.to_datetime(df['date'])
    df['amount'] = df['amount'].astype(float)
    df['month'] = df['date'].dt.to_period('M')
    
    monthly_expense = df.groupby('month')['amount'].sum().reset_index()
    return monthly_expense, df

def predict_future_expenses(user_id):
    transactions = get_user_transactions(user_id)
    if len(transactions) < 3:
        return "Not enough data for prediction."
    
    df = pd.DataFrame(transactions, columns=['date', 'amount'])
    df['date'] = pd.to_datetime(df['date'])
    df['amount'] = df['amount'].astype(float)
    df['days'] = (df['date'] - df['date'].min()).dt.days
    
    X = df[['days']]
    y = df['amount']
    model = LinearRegression()
    model.fit(X, y)
    
    future_days = max(df['days']) + 30
    future_expense = model.predict(np.array([[future_days]]))[0]
    
    return round(future_expense, 2)

@app.route('/ai_insights')
def ai_insights():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    user_id = session['user_id']
    monthly_expense, df = analyze_spending_pattern(user_id)
    predicted_expense = predict_future_expenses(user_id)
    
    if isinstance(monthly_expense, str):
        return monthly_expense
    
    plt.figure(figsize=(8, 5))
    plt.plot(monthly_expense['month'].astype(str), monthly_expense['amount'], marker='o', linestyle='-')
    plt.xlabel('Month')
    plt.ylabel('Total Expense')
    plt.title('Monthly Expense Trend')
    plt.xticks(rotation=45)
    plt.grid()
    chart_path = 'static/spending_trend.png'
    plt.savefig(chart_path)
    plt.close()
    
    return render_template('ai_insights.html', chart_path=chart_path, predicted_expense=predicted_expense)

if __name__ == "__main__":
    try:
       app.run(host="0.0.0.0", port=5001)
    except SystemExit as e:
        print(f"Flask exited with code: {e.code}")

