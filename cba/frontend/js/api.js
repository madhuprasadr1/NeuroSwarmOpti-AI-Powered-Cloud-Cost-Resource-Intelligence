/**
 * CloudOpt AI — API Client Layer
 * Handles async communication with the FastAPI backend.
 */

const API_BASE = '/api';

function extractErrorMessage(data, fallback = 'Operation failed') {
  if (!data) return fallback;
  if (typeof data === 'string') return data;
  if (typeof data.message === 'string' && data.message.trim()) return data.message;
  if (typeof data.detail === 'string' && data.detail.trim()) return data.detail;
  if (Array.isArray(data.detail)) {
    return data.detail.map(d => {
      const loc = Array.isArray(d.loc) ? d.loc.filter(x => x !== 'body').join('.') : '';
      return loc ? `${loc}: ${d.msg}` : (d.msg || JSON.stringify(d));
    }).join('; ');
  }
  if (typeof data.detail === 'object' && data.detail !== null) {
    return JSON.stringify(data.detail);
  }
  return fallback;
}

export const Api = {
  async getStatus() {
    const res = await fetch(`${API_BASE}/status`);
    if (!res.ok) throw new Error(await res.text());
    return res.json();
  },

  async getRealAnalytics() {
    const res = await fetch(`${API_BASE}/analytics/real-metrics`);
    if (!res.ok) throw new Error(await res.text());
    return res.json();
  },

  async getRLPolicyMetrics() {
    const res = await fetch(`${API_BASE}/rl/policy-metrics`);
    if (!res.ok) throw new Error(await res.text());
    return res.json();
  },

  async validateAuth(payload) {
    const res = await fetch(`${API_BASE}/auth/validate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(extractErrorMessage(data, 'Validation failed'));
    return data;
  },

  async getAwsSessionToken(payload) {
    const res = await fetch(`${API_BASE}/auth/aws/session-token`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload || {}),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(extractErrorMessage(data, 'Failed to acquire AWS session token'));
    return data;
  },

  async autoAttachAwsPolicies(payload) {
    const res = await fetch(`${API_BASE}/auth/aws/auto-attach-policies`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(extractErrorMessage(data, 'Auto-attach policies failed'));
    return data;
  },

  async fetchTelemetry(payload) {
    const res = await fetch(`${API_BASE}/telemetry/fetch`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || data.message || 'Telemetry fetch failed');
    return data;
  },

  async getTelemetry() {
    const res = await fetch(`${API_BASE}/telemetry`);
    if (!res.ok) throw new Error(await res.text());
    return res.json();
  },

  async generateRecommendations() {
    const res = await fetch(`${API_BASE}/recommendations/generate`, {
      method: 'POST',
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Inference generation failed');
    return data;
  },

  async getRecommendations() {
    const res = await fetch(`${API_BASE}/recommendations`);
    if (!res.ok) throw new Error(await res.text());
    return res.json();
  },

  async getExplainability(resourceId) {
    const res = await fetch(`${API_BASE}/explainability/${encodeURIComponent(resourceId)}`);
    if (!res.ok) throw new Error(await res.text());
    return res.json();
  },

  async dispatchQueue(payload) {
    const res = await fetch(`${API_BASE}/queue/dispatch`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(extractErrorMessage(data, 'Queue dispatch failed'));
    return data;
  },

  async executeAction(payload) {
    const res = await fetch(`${API_BASE}/actions/execute`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(extractErrorMessage(data, 'Action execution failed'));
    return data;
  },

  async autoOptimizeAws(payload = {}) {
    const res = await fetch(`${API_BASE}/aws/auto-optimize`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(extractErrorMessage(data, 'AWS Auto-optimization failed'));
    return data;
  },

  async tagAwsRecommendation(payload = {}) {
    const res = await fetch(`${API_BASE}/aws/tag-recommendation`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(extractErrorMessage(data, 'AWS Tagging failed'));
    return data;
  },

  async getQueueMessages() {
    const res = await fetch(`${API_BASE}/queue/messages`);
    if (!res.ok) throw new Error(await res.text());
    return res.json();
  },

  async getRollbackActions() {
    const res = await fetch(`${API_BASE}/rollback/actions`);
    if (!res.ok) throw new Error(await res.text());
    return res.json();
  },

  async stepRollback(payload) {
    const res = await fetch(`${API_BASE}/rollback/step`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Health step evaluation failed');
    return data;
  },

  async overrideRollback(payload) {
    const res = await fetch(`${API_BASE}/rollback/override`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Rollback override failed');
    return data;
  },

  async calculateForecast(payload) {
    const res = await fetch(`${API_BASE}/forecast`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Forecast calculation failed');
    return data;
  },

  downloadCloudFormationUrl() {
    return `${API_BASE}/download/cloudformation`;
  },

  async getUserSession() {
    const res = await fetch(`${API_BASE}/user/session`);
    if (!res.ok) throw new Error(await res.text());
    return res.json();
  },

  async loginUser(payload) {
    const res = await fetch(`${API_BASE}/user/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(extractErrorMessage(data, 'Sign in failed'));
    return data;
  },

  async logoutUser() {
    const res = await fetch(`${API_BASE}/user/logout`, {
      method: 'POST',
    });
    const data = await res.json();
    if (!res.ok) throw new Error(extractErrorMessage(data, 'Sign out failed'));
    return data;
  },

  async getNotificationHistory(limit = 50) {
    const res = await fetch(`${API_BASE}/notifications/history?limit=${limit}`);
    if (!res.ok) throw new Error(await res.text());
    return res.json();
  },

  getNotificationPreviewUrl(alertId) {
    return `${API_BASE}/notifications/preview/${encodeURIComponent(alertId)}`;
  },

  async sendTestAlert() {
    const res = await fetch(`${API_BASE}/notifications/test-email`, {
      method: 'POST',
    });
    const data = await res.json();
    if (!res.ok) throw new Error(extractErrorMessage(data, 'Test email dispatch failed'));
    return data;
  },

  async getSmtpConfig() {
    const res = await fetch(`${API_BASE}/notifications/smtp-config`);
    if (!res.ok) throw new Error(await res.text());
    return res.json();
  },

  async saveSmtpConfig(payload) {
    const res = await fetch(`${API_BASE}/notifications/smtp-config`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(extractErrorMessage(data, 'Saving SMTP sender config failed'));
    return data;
  },
};

