import React, { useState } from "react";
import api from "./api";
import { useNavigate, Link } from "react-router-dom";
import "./auth.css";

function Register() {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const navigate = useNavigate();

  const handleRegister = async () => {
    if (!username || !password) return alert("Enter username and password");
    const { data } = await api.post("/api/register", { username, password });
    if (data.status === "success") {
      alert("Registered");
      navigate("/");
    } else {
      alert(data.message || "Registration failed");
    }
  };

  return (
    <div className="auth-container">
      <div className="auth-card">
        <div className="auth-header">
          <h2 className="auth-title register">Register</h2>
        </div>
        <input
          className="auth-input"
          placeholder="Username"
          value={username}
          onChange={(e) => setUsername(e.target.value)}
        />
        <input
          className="auth-input"
          type="password"
          placeholder="Password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
        />
        <button className="auth-button" onClick={handleRegister}>
          Create account
        </button>
        <p style={{ marginTop: "1rem", textAlign: "center" }}>
          Have an account? <Link to="/">Login</Link>
        </p>
      </div>
    </div>
  );
}

export default Register;
