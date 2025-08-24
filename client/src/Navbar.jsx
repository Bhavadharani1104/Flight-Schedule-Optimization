import React from 'react';
import { Link } from 'react-router-dom';
import './navbar.css';

export default function Navbar({ onLogout }) {
  return (
    <nav className="aviation-navbar">
      <div className="container">
        <Link className="aviation-brand" to="/dashboard">
          Flight Scheduler
        </Link>
        <div className="aviation-nav-links">
          <Link className="aviation-nav-btn" to="/dashboard">Dashboard</Link>
          <Link className="aviation-nav-btn" to="/analysis">Analysis</Link>
          <Link className="aviation-nav-btn" to="/chatbot">Chatbot</Link>
          <button className="aviation-logout-btn" onClick={onLogout}>
            Logout
          </button>
        </div>
      </div>
    </nav>
  );
}
