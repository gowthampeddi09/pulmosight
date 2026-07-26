'use client';

import { useState, useCallback, useRef, useEffect } from 'react';
import {
  Activity, Shield, FileText, Upload, History, Moon, Sun,
  ArrowRight, AlertCircle, RefreshCw, Trash2, ChevronLeft,
  ChevronRight, Search, Download, LogOut,
} from 'lucide-react';
import { api } from '@/lib/api';

type View = 'upload' | 'history';

export default function Home() {
  // Auth state
  const [token, setToken] = useState<string | null>(null);
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [fullName, setFullName] = useState('');
  const [isRegister, setIsRegister] = useState(false);

  // Theme
  const [darkMode, setDarkMode] = useState(true);

  // Current view
  const [view, setView] = useState<View>('upload');

  // Upload state
  const [file, setFile] = useState<File | null>(null);
  const [filePreview, setFilePreview] = useState<string | null>(null);
  const [prediction, setPrediction] = useState<any>(null);
  const [report, setReport] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Drag-and-drop state
  const [isDragOver, setIsDragOver] = useState(false);
  const dropRef = useRef<HTMLDivElement>(null);

  // History state
  const [historyItems, setHistoryItems] = useState<any[]>([]);
  const [historyTotal, setHistoryTotal] = useState(0);
  const [historyPage, setHistoryPage] = useState(1);
  const [historyTotalPages, setHistoryTotalPages] = useState(0);
  const [historySearch, setHistorySearch] = useState('');
  const [historyLabel, setHistoryLabel] = useState('');
  const [historySortOrder, setHistorySortOrder] = useState('desc');
  const [historyLoading, setHistoryLoading] = useState(false);

  // Patient context for report
  const [patientAge, setPatientAge] = useState('');
  const [patientGender, setPatientGender] = useState('');
  const [patientSymptoms, setPatientSymptoms] = useState('');

  // Restore token from localStorage on mount
  useEffect(() => {
    const saved = localStorage.getItem('access_token');
    if (saved) setToken(saved);
  }, []);

  // Toggle dark mode on html element
  useEffect(() => {
    document.documentElement.classList.toggle('dark', darkMode);
  }, [darkMode]);

  const apiBase = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

  // ---------- Auth ----------
  const handleAuth = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const endpoint = isRegister ? '/auth/register' : '/auth/login';
      const body = isRegister ? { email, password, full_name: fullName } : { email, password };
      const res = await api.post(endpoint, body);
      const accessToken = res.data.access_token;
      setToken(accessToken);
      localStorage.setItem('access_token', accessToken);
    } catch (err: any) {
      setError(err.response?.data?.detail?.error?.message || err.response?.data?.error?.message || 'Authentication failed');
    } finally {
      setLoading(false);
    }
  };

  const handleLogout = () => {
    setToken(null);
    localStorage.removeItem('access_token');
    setPrediction(null);
    setReport(null);
    setFile(null);
    setFilePreview(null);
  };

  // ---------- File handling with drag-and-drop ----------
  const handleFile = useCallback((f: File) => {
    const valid = ['image/jpeg', 'image/png'];
    if (!valid.includes(f.type)) {
      setError('Only JPEG and PNG images are accepted');
      return;
    }
    if (f.size > 10 * 1024 * 1024) {
      setError('File must be under 10 MB');
      return;
    }
    setFile(f);
    setError(null);
    setPrediction(null);
    setReport(null);
    const reader = new FileReader();
    reader.onloadend = () => setFilePreview(reader.result as string);
    reader.readAsDataURL(f);
  }, []);

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    const dropped = e.dataTransfer.files?.[0];
    if (dropped) handleFile(dropped);
  }, [handleFile]);

  const onDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  }, []);

  const onDragLeave = useCallback(() => setIsDragOver(false), []);

  // ---------- Predict ----------
  const handlePredict = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    setPrediction(null);
    setReport(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await api.post('/predict', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      setPrediction(res.data);
    } catch (err: any) {
      setError(err.response?.data?.detail?.error?.message || err.response?.data?.error?.message || 'Inference failed');
    } finally {
      setLoading(false);
    }
  };

  // ---------- Generate Report ----------
  const handleGenerateReport = async () => {
    if (!prediction) return;
    setLoading(true);
    setError(null);
    try {
      const body: any = { prediction_id: prediction.id };
      if (patientAge) body.patient_age = parseInt(patientAge);
      if (patientGender) body.patient_gender = patientGender;
      if (patientSymptoms) body.patient_symptoms = patientSymptoms;

      const res = await api.post('/generate-report', body);
      setReport(res.data.report_text);
    } catch (err: any) {
      setError(err.response?.data?.detail?.error?.message || err.response?.data?.error?.message || 'Report generation failed');
    } finally {
      setLoading(false);
    }
  };

  // ---------- PDF Download ----------
  const handleDownloadPDF = async () => {
    if (!prediction) return;
    try {
      const res = await api.get(`/prediction/${prediction.id}/pdf`, {
        responseType: 'blob',
      });
      const url = window.URL.createObjectURL(new Blob([res.data]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', `pulmosight_report_${prediction.id}.pdf`);
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.URL.revokeObjectURL(url);
    } catch (err: any) {
      setError('PDF download failed');
    }
  };

  // ---------- History ----------
  const fetchHistory = useCallback(async (page = 1) => {
    setHistoryLoading(true);
    try {
      const params: any = { page, per_page: 10, sort_order: historySortOrder };
      if (historySearch) params.search = historySearch;
      if (historyLabel) params.label = historyLabel;
      const res = await api.get('/history', { params });
      setHistoryItems(res.data.items);
      setHistoryTotal(res.data.total);
      setHistoryPage(res.data.page);
      setHistoryTotalPages(res.data.total_pages);
    } catch (err: any) {
      setError('Failed to load history');
    } finally {
      setHistoryLoading(false);
    }
  }, [historySearch, historyLabel, historySortOrder]);

  useEffect(() => {
    if (token && view === 'history') fetchHistory(1);
  }, [token, view, fetchHistory]);

  const handleDeletePrediction = async (id: string) => {
    if (!confirm('Delete this prediction permanently?')) return;
    try {
      await api.delete(`/prediction/${id}`);
      fetchHistory(historyPage);
    } catch {
      setError('Delete failed');
    }
  };

  // ---------- Confidence gauge ----------
  const ConfidenceGauge = ({ value }: { value: number }) => {
    const pct = Math.round(value * 100);
    const radius = 40;
    const circumference = 2 * Math.PI * radius;
    const offset = circumference - (value * circumference);
    return (
      <div className="relative w-24 h-24 mx-auto">
        <svg className="w-24 h-24 -rotate-90" viewBox="0 0 100 100">
          <circle cx="50" cy="50" r={radius} stroke="currentColor" strokeWidth="8" fill="none" className="text-slate-800" />
          <circle cx="50" cy="50" r={radius} stroke="currentColor" strokeWidth="8" fill="none"
            className={pct >= 80 ? 'text-red-500' : pct >= 60 ? 'text-amber-500' : 'text-emerald-500'}
            strokeDasharray={circumference} strokeDashoffset={offset}
            strokeLinecap="round" style={{ transition: 'stroke-dashoffset 0.6s ease' }}
          />
        </svg>
        <span className="absolute inset-0 flex items-center justify-center text-lg font-bold">{pct}%</span>
      </div>
    );
  };

  // ====================================================================
  // RENDER
  // ====================================================================
  return (
    <div className={`min-h-screen flex flex-col font-sans transition-colors duration-200 ${darkMode ? 'bg-slate-950 text-slate-100' : 'bg-slate-50 text-slate-900'}`}>
      {/* Header */}
      <header className={`border-b sticky top-0 z-50 px-6 py-3 flex items-center justify-between backdrop-blur ${darkMode ? 'border-slate-800 bg-slate-900/80' : 'border-slate-200 bg-white/80'}`}>
        <div className="flex items-center space-x-3">
          <Activity className="h-7 w-7 text-sky-500" />
          <span className="text-xl font-bold tracking-tight bg-gradient-to-r from-sky-400 to-indigo-400 bg-clip-text text-transparent">
            PulmoSight AI
          </span>
        </div>
        <div className="flex items-center gap-3">
          {token && (
            <>
              <button
                onClick={() => setView(view === 'upload' ? 'history' : 'upload')}
                className={`text-xs px-3 py-1.5 rounded-lg border transition-colors ${darkMode ? 'border-slate-700 text-slate-300 hover:text-white' : 'border-slate-300 text-slate-600 hover:text-slate-900'}`}
              >
                {view === 'upload' ? <span className="flex items-center gap-1.5"><History className="h-3.5 w-3.5" /> History</span>
                  : <span className="flex items-center gap-1.5"><Upload className="h-3.5 w-3.5" /> Upload</span>}
              </button>
              <button onClick={handleLogout} className={`text-xs px-3 py-1.5 rounded-lg border transition-colors ${darkMode ? 'border-slate-700 text-slate-400 hover:text-slate-200' : 'border-slate-300 text-slate-500 hover:text-slate-800'}`}>
                <LogOut className="h-3.5 w-3.5" />
              </button>
            </>
          )}
          <button onClick={() => setDarkMode(!darkMode)} className="p-1.5 rounded-lg hover:bg-slate-800/50 transition-colors" aria-label="Toggle dark mode">
            {darkMode ? <Sun className="h-4 w-4 text-amber-400" /> : <Moon className="h-4 w-4 text-indigo-500" />}
          </button>
        </div>
      </header>

      {/* Main */}
      <main className="flex-1 max-w-7xl w-full mx-auto p-6">
        {!token ? (
          /* =================== AUTH FORM =================== */
          <div className={`max-w-md mx-auto my-12 p-8 rounded-2xl shadow-2xl border ${darkMode ? 'bg-slate-900 border-slate-800' : 'bg-white border-slate-200'}`}>
            <div className="text-center mb-6">
              <Shield className="h-10 w-10 text-sky-500 mx-auto mb-2" />
              <h2 className="text-2xl font-bold">{isRegister ? 'Create Account' : 'Clinician Portal'}</h2>
              <p className={`text-sm mt-1 ${darkMode ? 'text-slate-400' : 'text-slate-500'}`}>Chest X-Ray AI Diagnostics Platform</p>
            </div>
            {error && (
              <div className="mb-4 p-3 bg-red-950/50 border border-red-800 text-red-300 rounded-lg text-sm flex items-center gap-2">
                <AlertCircle className="h-4 w-4 shrink-0" />
                <span>{error}</span>
              </div>
            )}
            <form onSubmit={handleAuth} className="space-y-4">
              {isRegister && (
                <div>
                  <label className={`block text-xs font-semibold uppercase mb-1 ${darkMode ? 'text-slate-400' : 'text-slate-500'}`}>Full Name</label>
                  <input type="text" required value={fullName} onChange={(e) => setFullName(e.target.value)} placeholder="Dr. Jane Doe"
                    className={`w-full rounded-lg px-4 py-2.5 text-sm border focus:outline-none focus:border-sky-500 ${darkMode ? 'bg-slate-950 border-slate-800' : 'bg-slate-50 border-slate-300'}`} />
                </div>
              )}
              <div>
                <label className={`block text-xs font-semibold uppercase mb-1 ${darkMode ? 'text-slate-400' : 'text-slate-500'}`}>Email</label>
                <input type="email" required value={email} onChange={(e) => setEmail(e.target.value)} placeholder="doctor@hospital.com"
                  className={`w-full rounded-lg px-4 py-2.5 text-sm border focus:outline-none focus:border-sky-500 ${darkMode ? 'bg-slate-950 border-slate-800' : 'bg-slate-50 border-slate-300'}`} />
              </div>
              <div>
                <label className={`block text-xs font-semibold uppercase mb-1 ${darkMode ? 'text-slate-400' : 'text-slate-500'}`}>Password</label>
                <input type="password" required value={password} onChange={(e) => setPassword(e.target.value)} placeholder="Min. 8 characters"
                  className={`w-full rounded-lg px-4 py-2.5 text-sm border focus:outline-none focus:border-sky-500 ${darkMode ? 'bg-slate-950 border-slate-800' : 'bg-slate-50 border-slate-300'}`} />
              </div>
              <button type="submit" disabled={loading}
                className="w-full bg-sky-500 hover:bg-sky-400 text-white font-semibold py-2.5 rounded-lg transition-colors flex items-center justify-center gap-2 disabled:opacity-50">
                {loading ? <RefreshCw className="h-4 w-4 animate-spin" /> : (isRegister ? 'Register' : 'Sign In')}
              </button>
            </form>
            <div className="mt-4 text-center">
              <button onClick={() => { setIsRegister(!isRegister); setError(null); }} className="text-xs text-sky-400 hover:underline">
                {isRegister ? 'Already have an account? Sign in' : "Don't have an account? Register"}
              </button>
            </div>
          </div>
        ) : view === 'upload' ? (
          /* =================== UPLOAD + RESULTS =================== */
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
            {/* Left: Upload + Patient Context */}
            <div className="lg:col-span-5 space-y-6">
              <div className={`p-6 rounded-2xl border ${darkMode ? 'bg-slate-900 border-slate-800' : 'bg-white border-slate-200'}`}>
                <h3 className="font-semibold text-lg mb-4 flex items-center gap-2">
                  <Upload className="h-5 w-5 text-sky-500" />
                  Chest X-Ray Upload
                </h3>
                {error && (
                  <div className="mb-4 p-3 bg-red-950/50 border border-red-800 text-red-300 rounded-lg text-sm flex items-center gap-2">
                    <AlertCircle className="h-4 w-4 shrink-0" /><span>{error}</span>
                  </div>
                )}
                {/* Drag-and-drop zone */}
                <div
                  ref={dropRef}
                  onDrop={onDrop}
                  onDragOver={onDragOver}
                  onDragLeave={onDragLeave}
                  className={`border-2 border-dashed rounded-xl p-8 text-center transition-colors cursor-pointer ${isDragOver
                    ? 'border-sky-500 bg-sky-500/10'
                    : darkMode ? 'border-slate-800 hover:border-sky-500/50 bg-slate-950/50' : 'border-slate-300 hover:border-sky-400 bg-slate-50'
                  }`}
                >
                  <input type="file" accept="image/jpeg,image/png" onChange={(e) => { if (e.target.files?.[0]) handleFile(e.target.files[0]); }}
                    className="hidden" id="xray-file" />
                  <label htmlFor="xray-file" className="cursor-pointer block">
                    {filePreview ? (
                      <img src={filePreview} alt="Preview" className="max-h-40 mx-auto rounded-lg mb-2 object-contain" />
                    ) : (
                      <FileText className={`h-10 w-10 mx-auto mb-2 ${darkMode ? 'text-slate-600' : 'text-slate-400'}`} />
                    )}
                    <span className={`text-sm font-medium ${darkMode ? 'text-slate-300' : 'text-slate-600'}`}>
                      {file ? file.name : 'Drag & drop or click to select a Chest X-Ray (JPEG/PNG)'}
                    </span>
                  </label>
                </div>
                <button onClick={handlePredict} disabled={!file || loading}
                  className="w-full mt-4 bg-sky-500 hover:bg-sky-400 disabled:opacity-50 text-white font-semibold py-2.5 rounded-lg transition-colors flex items-center justify-center gap-2">
                  {loading ? <RefreshCw className="h-4 w-4 animate-spin" /> : <><ArrowRight className="h-4 w-4" /> Run AI Analysis</>}
                </button>
              </div>

              {/* Patient Context (optional) */}
              {prediction && !report && (
                <div className={`p-6 rounded-2xl border ${darkMode ? 'bg-slate-900 border-slate-800' : 'bg-white border-slate-200'}`}>
                  <h4 className="font-semibold text-sm mb-3 text-sky-400">Optional Patient Context (for LLM report)</h4>
                  <div className="space-y-3">
                    <input type="number" placeholder="Age" value={patientAge} onChange={(e) => setPatientAge(e.target.value)}
                      className={`w-full rounded-lg px-3 py-2 text-sm border focus:outline-none focus:border-sky-500 ${darkMode ? 'bg-slate-950 border-slate-800' : 'bg-slate-50 border-slate-300'}`} />
                    <select value={patientGender} onChange={(e) => setPatientGender(e.target.value)}
                      className={`w-full rounded-lg px-3 py-2 text-sm border focus:outline-none focus:border-sky-500 ${darkMode ? 'bg-slate-950 border-slate-800' : 'bg-slate-50 border-slate-300'}`}>
                      <option value="">Gender (optional)</option>
                      <option value="Male">Male</option>
                      <option value="Female">Female</option>
                      <option value="Other">Other</option>
                    </select>
                    <textarea placeholder="Presenting symptoms (e.g., cough, fever, dyspnea)" value={patientSymptoms} onChange={(e) => setPatientSymptoms(e.target.value)} rows={2}
                      className={`w-full rounded-lg px-3 py-2 text-sm border focus:outline-none focus:border-sky-500 ${darkMode ? 'bg-slate-950 border-slate-800' : 'bg-slate-50 border-slate-300'}`} />
                  </div>
                </div>
              )}
            </div>

            {/* Right: Results */}
            <div className="lg:col-span-7 space-y-6">
              {prediction ? (
                <div className={`p-6 rounded-2xl border space-y-6 ${darkMode ? 'bg-slate-900 border-slate-800' : 'bg-white border-slate-200'}`}>
                  {/* Classification + Confidence */}
                  <div className="flex items-start justify-between pb-4 border-b border-slate-800/50">
                    <div>
                      <span className={`text-xs uppercase tracking-wider ${darkMode ? 'text-slate-400' : 'text-slate-500'}`}>Classification</span>
                      <div className={`text-3xl font-black ${prediction.prediction === 'PNEUMONIA' ? 'text-red-400' : 'text-emerald-400'}`}>
                        {prediction.prediction}
                      </div>
                      <span className={`text-xs ${darkMode ? 'text-slate-500' : 'text-slate-400'}`}>
                        {prediction.processing_time_ms?.toFixed(0)}ms · v{prediction.model_version}
                      </span>
                    </div>
                    <div className="text-center">
                      <span className={`text-xs uppercase tracking-wider ${darkMode ? 'text-slate-400' : 'text-slate-500'}`}>Confidence</span>
                      <ConfidenceGauge value={prediction.confidence} />
                    </div>
                  </div>

                  {/* Grad-CAM heatmap */}
                  {prediction.heatmap_url && (
                    <div>
                      <h4 className="text-sm font-semibold mb-2 text-sky-400">Grad-CAM Explainability Heatmap</h4>
                      <div className={`rounded-xl overflow-hidden border flex justify-center ${darkMode ? 'border-slate-800 bg-black' : 'border-slate-200 bg-slate-100'}`}>
                        <img
                          src={`${apiBase}${prediction.heatmap_url}`}
                          alt="Grad-CAM heatmap overlay"
                          className="max-h-64 object-contain"
                          onError={(e) => { (e.target as HTMLImageElement).style.display = 'none'; }}
                        />
                      </div>
                      {prediction.gradcam_observation && (
                        <p className={`text-xs italic mt-2 ${darkMode ? 'text-slate-400' : 'text-slate-500'}`}>{prediction.gradcam_observation}</p>
                      )}
                    </div>
                  )}

                  {/* Action buttons */}
                  <div className="pt-2 flex flex-wrap gap-3">
                    {!report ? (
                      <button onClick={handleGenerateReport} disabled={loading}
                        className="bg-indigo-600 hover:bg-indigo-500 text-white font-medium px-4 py-2 rounded-lg text-sm transition-colors flex items-center gap-2 disabled:opacity-50">
                        {loading ? <RefreshCw className="h-4 w-4 animate-spin" /> : <><FileText className="h-4 w-4" /> Generate Clinical Report</>}
                      </button>
                    ) : (
                      <button onClick={handleDownloadPDF}
                        className="bg-emerald-600 hover:bg-emerald-500 text-white font-medium px-4 py-2 rounded-lg text-sm transition-colors flex items-center gap-2">
                        <Download className="h-4 w-4" /> Download PDF Report
                      </button>
                    )}
                  </div>

                  {/* LLM Report */}
                  {report && (
                    <div className={`p-4 rounded-xl border text-sm leading-relaxed whitespace-pre-line max-h-80 overflow-y-auto ${darkMode ? 'bg-slate-950 border-slate-800 text-slate-300' : 'bg-slate-50 border-slate-200 text-slate-700'}`}>
                      {report}
                    </div>
                  )}
                </div>
              ) : (
                <div className={`h-full border rounded-2xl p-12 flex flex-col items-center justify-center text-center ${darkMode ? 'border-slate-800 text-slate-500' : 'border-slate-200 text-slate-400'}`}>
                  <Activity className="h-12 w-12 stroke-1 mb-3 opacity-50" />
                  <p className="text-sm font-medium">Upload a Chest X-ray image to run the automated AI diagnostic pipeline</p>
                  <p className="text-xs mt-2 opacity-60">Supports JPEG and PNG up to 10 MB</p>
                </div>
              )}
            </div>
          </div>
        ) : (
          /* =================== HISTORY PAGE =================== */
          <div className="space-y-6">
            <div className="flex flex-wrap items-center gap-4">
              <h2 className="text-xl font-bold flex items-center gap-2">
                <History className="h-5 w-5 text-sky-500" /> Prediction History
              </h2>
              <div className="flex-1" />
              <div className="flex gap-2">
                <div className="relative">
                  <Search className={`absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 ${darkMode ? 'text-slate-500' : 'text-slate-400'}`} />
                  <input type="text" placeholder="Search..." value={historySearch}
                    onChange={(e) => setHistorySearch(e.target.value)}
                    onKeyDown={(e) => e.key === 'Enter' && fetchHistory(1)}
                    className={`pl-9 pr-3 py-2 rounded-lg text-sm border focus:outline-none focus:border-sky-500 w-48 ${darkMode ? 'bg-slate-900 border-slate-800' : 'bg-white border-slate-300'}`} />
                </div>
                <select value={historyLabel} onChange={(e) => { setHistoryLabel(e.target.value); }}
                  className={`rounded-lg px-3 py-2 text-sm border focus:outline-none focus:border-sky-500 ${darkMode ? 'bg-slate-900 border-slate-800' : 'bg-white border-slate-300'}`}>
                  <option value="">All Labels</option>
                  <option value="PNEUMONIA">Pneumonia</option>
                  <option value="NORMAL">Normal</option>
                </select>
                <select value={historySortOrder} onChange={(e) => { setHistorySortOrder(e.target.value); }}
                  className={`rounded-lg px-3 py-2 text-sm border focus:outline-none focus:border-sky-500 ${darkMode ? 'bg-slate-900 border-slate-800' : 'bg-white border-slate-300'}`}>
                  <option value="desc">Newest first</option>
                  <option value="asc">Oldest first</option>
                </select>
                <button onClick={() => fetchHistory(1)} className="bg-sky-500 hover:bg-sky-400 text-white px-3 py-2 rounded-lg text-sm font-medium">
                  Filter
                </button>
              </div>
            </div>

            {historyLoading ? (
              <div className="text-center py-12"><RefreshCw className="h-6 w-6 animate-spin mx-auto text-sky-500" /></div>
            ) : historyItems.length === 0 ? (
              <div className={`text-center py-12 rounded-2xl border ${darkMode ? 'border-slate-800 text-slate-500' : 'border-slate-200 text-slate-400'}`}>
                <p className="text-sm">No predictions found</p>
              </div>
            ) : (
              <div className="space-y-3">
                {historyItems.map((item: any) => (
                  <div key={item.id} className={`p-4 rounded-xl border flex items-center gap-4 group ${darkMode ? 'bg-slate-900 border-slate-800 hover:border-slate-700' : 'bg-white border-slate-200 hover:border-slate-300'} transition-colors`}>
                    <div className={`w-16 h-16 rounded-lg flex items-center justify-center text-xs font-bold ${item.prediction === 'PNEUMONIA' ? 'bg-red-500/20 text-red-400' : 'bg-emerald-500/20 text-emerald-400'}`}>
                      {item.prediction === 'PNEUMONIA' ? 'PNE' : 'NRM'}
                    </div>
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-semibold truncate">{item.filename}</p>
                      <p className={`text-xs ${darkMode ? 'text-slate-400' : 'text-slate-500'}`}>
                        {new Date(item.created_at).toLocaleString()} · {(item.confidence * 100).toFixed(1)}% confidence
                      </p>
                    </div>
                    <div className={`text-sm font-bold ${item.prediction === 'PNEUMONIA' ? 'text-red-400' : 'text-emerald-400'}`}>
                      {item.prediction}
                    </div>
                    <button onClick={() => handleDeletePrediction(item.id)}
                      className="opacity-0 group-hover:opacity-100 text-red-400 hover:text-red-300 p-2 rounded-lg transition-opacity" title="Delete">
                      <Trash2 className="h-4 w-4" />
                    </button>
                  </div>
                ))}

                {/* Pagination */}
                {historyTotalPages > 1 && (
                  <div className="flex items-center justify-center gap-3 pt-4">
                    <button disabled={historyPage <= 1} onClick={() => fetchHistory(historyPage - 1)}
                      className="p-2 rounded-lg border border-slate-700 disabled:opacity-30 hover:bg-slate-800 transition-colors">
                      <ChevronLeft className="h-4 w-4" />
                    </button>
                    <span className="text-sm">Page {historyPage} of {historyTotalPages} ({historyTotal} total)</span>
                    <button disabled={historyPage >= historyTotalPages} onClick={() => fetchHistory(historyPage + 1)}
                      className="p-2 rounded-lg border border-slate-700 disabled:opacity-30 hover:bg-slate-800 transition-colors">
                      <ChevronRight className="h-4 w-4" />
                    </button>
                  </div>
                )}
              </div>
            )}
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className={`text-center py-4 text-xs border-t ${darkMode ? 'border-slate-800 text-slate-500' : 'border-slate-200 text-slate-400'}`}>
        PulmoSight AI — Decision-support tool for educational purposes only. Not a medical device.
      </footer>
    </div>
  );
}
