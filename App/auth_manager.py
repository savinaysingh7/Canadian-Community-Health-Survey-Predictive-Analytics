import hashlib
import os
import database_manager as db

def hash_password(password):
    """Simple SHA-256 hashing (In production, use bcrypt/argon2 with salt)."""
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(stored_hash, provided_password):
    return stored_hash == hashlib.sha256(provided_password.encode()).hexdigest()

def check_login(username, password):
    user = db.get_user(username)
    if user and verify_password(user['password_hash'], password):
        return user
    return None

def init_auth():
    """Ensure default admin exists."""
    if not db.get_user("admin"):
        db.create_user("admin", hash_password("admin123"), "admin")
