import React, { useState, useCallback } from 'react';
import { Upload, Download, BarChart3, TrendingUp, AlertTriangle, CheckCircle, FileText, Github as GitHub } from 'lucide-react';

interface ScoreResult {
  wallet: string;
  score: number;
  risk_level: string;
  features: Record<string, number>;
}

interface ScoreDistribution {
  range: string;
  count: number;
  percentage: number;
}

function CreditScoring() {
  const [file, setFile] = useState<File | null>(null);
  const [results, setResults] = useState<ScoreResult[]>([]);
  const [distribution, setDistribution] = useState<ScoreDistribution[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleFileUpload = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    const uploadedFile = event.target.files?.[0];
    if (uploadedFile) {
      setFile(uploadedFile);
      setError(null);
    }
  }, []);

  const processFile = useCallback(async () => {
    if (!file) return;

    setLoading(true);
    setError(null);

    try {
      const text = await file.text();
      const transactions = JSON.parse(text);
      
      // Mock processing - in real implementation, this would call the Python backend
      const mockResults: ScoreResult[] = [
        {
          wallet: "0x1234...5678",
          score: 785,
          risk_level: "Low",
          features: {
            repayment_ratio: 0.98,
            liquidation_count: 0,
            transaction_frequency: 0.75,
            deposit_consistency: 0.85,
            avg_health_factor: 2.1
          }
        },
        {
          wallet: "0x9876...5432",
          score: 342,
          risk_level: "High",
          features: {
            repayment_ratio: 0.65,
            liquidation_count: 3,
            transaction_frequency: 0.45,
            deposit_consistency: 0.35,
            avg_health_factor: 1.2
          }
        },
        {
          wallet: "0xabcd...ef01",
          score: 623,
          risk_level: "Medium",
          features: {
            repayment_ratio: 0.88,
            liquidation_count: 1,
            transaction_frequency: 0.60,
            deposit_consistency: 0.70,
            avg_health_factor: 1.8
          }
        }
      ];

      const mockDistribution: ScoreDistribution[] = [
        { range: "0-100", count: 1250, percentage: 12.5 },
        { range: "100-200", count: 1680, percentage: 16.8 },
        { range: "200-300", count: 2340, percentage: 23.4 },
        { range: "300-400", count: 1890, percentage: 18.9 },
        { range: "400-500", count: 1245, percentage: 12.45 },
        { range: "500-600", count: 856, percentage: 8.56 },
        { range: "600-700", count: 445, percentage: 4.45 },
        { range: "700-800", count: 234, percentage: 2.34 },
        { range: "800-900", count: 45, percentage: 0.45 },
        { range: "900-1000", count: 15, percentage: 0.15 }
      ];

      setResults(mockResults);
      setDistribution(mockDistribution);
    } catch (err) {
      setError('Error processing file. Please ensure it\'s a valid JSON file.');
    } finally {
      setLoading(false);
    }
  }, [file]);

  const getRiskColor = (riskLevel: string) => {
    switch (riskLevel) {
      case 'Low': return 'text-green-600 bg-green-50';
      case 'Medium': return 'text-yellow-600 bg-yellow-50';
      case 'High': return 'text-red-600 bg-red-50';
      default: return 'text-gray-600 bg-gray-50';
    }
  };

  const getScoreColor = (score: number) => {
    if (score >= 700) return 'text-green-600';
    if (score >= 400) return 'text-yellow-600';
    return 'text-red-600';
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center">
              <TrendingUp className="h-8 w-8 text-blue-600 mr-3" />
              <h1 className="text-2xl font-bold text-gray-900">DeFi Credit Scoring</h1>
            </div>
            <div className="flex items-center space-x-4">
              <span className="text-sm text-gray-500">Aave V2 Protocol</span>
              <GitHub className="h-5 w-5 text-gray-400" />
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Upload Section */}
        <div className="bg-white rounded-lg shadow-sm p-6 mb-8">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">Upload Transaction Data</h2>
          <div className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center">
            <Upload className="mx-auto h-12 w-12 text-gray-400 mb-4" />
            <div className="space-y-2">
              <p className="text-sm text-gray-600">
                Upload your Aave V2 transaction data (JSON format)
              </p>
              <input
                type="file"
                accept=".json"
                onChange={handleFileUpload}
                className="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-md file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"
              />
            </div>
            {file && (
              <div className="mt-4 flex items-center justify-center space-x-4">
                <span className="text-sm text-gray-600">File: {file.name}</span>
                <button
                  onClick={processFile}
                  disabled={loading}
                  className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {loading ? 'Processing...' : 'Process File'}
                </button>
              </div>
            )}
          </div>
          {error && (
            <div className="mt-4 flex items-center space-x-2 text-red-600">
              <AlertTriangle className="h-5 w-5" />
              <span className="text-sm">{error}</span>
            </div>
          )}
        </div>

        {/* Results Section */}
        {results.length > 0 && (
          <>
            {/* Score Distribution */}
            <div className="bg-white rounded-lg shadow-sm p-6 mb-8">
              <h2 className="text-lg font-semibold text-gray-900 mb-4">Score Distribution</h2>
              <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
                {distribution.map((item) => (
                  <div key={item.range} className="bg-gray-50 rounded-lg p-4 text-center">
                    <div className="text-2xl font-bold text-blue-600">{item.count}</div>
                    <div className="text-sm text-gray-600">{item.range}</div>
                    <div className="text-xs text-gray-500">{item.percentage}%</div>
                  </div>
                ))}
              </div>
            </div>

            {/* Sample Results */}
            <div className="bg-white rounded-lg shadow-sm p-6">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-lg font-semibold text-gray-900">Sample Wallet Scores</h2>
                <button className="flex items-center space-x-2 px-4 py-2 bg-green-600 text-white rounded-md hover:bg-green-700">
                  <Download className="h-4 w-4" />
                  <span>Export Results</span>
                </button>
              </div>
              
              <div className="overflow-x-auto">
                <table className="min-w-full divide-y divide-gray-200">
                  <thead className="bg-gray-50">
                    <tr>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Wallet Address
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Credit Score
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Risk Level
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Key Features
                      </th>
                    </tr>
                  </thead>
                  <tbody className="bg-white divide-y divide-gray-200">
                    {results.map((result) => (
                      <tr key={result.wallet}>
                        <td className="px-6 py-4 whitespace-nowrap text-sm font-mono text-gray-900">
                          {result.wallet}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <div className={`text-2xl font-bold ${getScoreColor(result.score)}`}>
                            {result.score}
                          </div>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <span className={`inline-flex px-2 py-1 text-xs font-medium rounded-full ${getRiskColor(result.risk_level)}`}>
                            {result.risk_level}
                          </span>
                        </td>
                        <td className="px-6 py-4 text-sm text-gray-600">
                          <div className="space-y-1">
                            <div>Repayment: {(result.features.repayment_ratio * 100).toFixed(1)}%</div>
                            <div>Liquidations: {result.features.liquidation_count}</div>
                            <div>Health Factor: {result.features.avg_health_factor.toFixed(1)}</div>
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </>
        )}

        {/* Documentation Links */}
        <div className="mt-8 bg-white rounded-lg shadow-sm p-6">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">Documentation</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="flex items-center space-x-3 p-4 bg-blue-50 rounded-lg">
              <FileText className="h-8 w-8 text-blue-600" />
              <div>
                <h3 className="font-medium text-gray-900">README.md</h3>
                <p className="text-sm text-gray-600">Setup and usage instructions</p>
              </div>
            </div>
            <div className="flex items-center space-x-3 p-4 bg-green-50 rounded-lg">
              <BarChart3 className="h-8 w-8 text-green-600" />
              <div>
                <h3 className="font-medium text-gray-900">analysis.md</h3>
                <p className="text-sm text-gray-600">Detailed score analysis</p>
              </div>
            </div>
            <div className="flex items-center space-x-3 p-4 bg-purple-50 rounded-lg">
              <CheckCircle className="h-8 w-8 text-purple-600" />
              <div>
                <h3 className="font-medium text-gray-900">model_validation.py</h3>
                <p className="text-sm text-gray-600">Model validation results</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default CreditScoring;