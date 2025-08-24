import React, { useState } from "react";
import { BrowserRouter as Router, Routes, Route, Navigate } from "react-router-dom";
import Login from "./Login";
import Register from "./Register";
import Dashboard from "./Dashboard";
import Analysis from "./Analysis";
import Chatbot from "./Chatbot";
import Navbar from "./Navbar";
function App() {
  const [loggedIn, setLoggedIn] = useState(false);

  const handleLogout = () => setLoggedIn(false);

  return (
    <Router>
      {loggedIn && <Navbar onLogout={handleLogout} />}
      <Routes>
        <Route path="/" element={loggedIn ? <Navigate to="/dashboard" /> : <Login onLogin={setLoggedIn} />} />
        <Route path="/register" element={<Register />} />
        <Route path="/dashboard" element={loggedIn ? <Dashboard /> : <Navigate to="/" />} />
        <Route path="/analysis" element={loggedIn ? <Analysis /> : <Navigate to="/" />} />
        <Route path="/chatbot" element={loggedIn ? <Chatbot /> : <Navigate to="/" />} />
      </Routes>
    </Router>
  );
}
export default App;
