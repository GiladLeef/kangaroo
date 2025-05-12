'use client';

import { useEffect, useState } from 'react';
import Link from 'next/link';

type ClientStats = {
  address: string;
  dpCount: number;
  lastSeen: string;
  clientInfo: string;
  percentage: number;
};

type PoolStats = {
  totalDP: number;
  clients: ClientStats[];
};

export default function PoolPage() {
  const [stats, setStats] = useState<PoolStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchStats = async () => {
    try {
      setLoading(true);
      const response = await fetch('/api/pool/stats');
      if (!response.ok) {
        throw new Error('Failed to fetch pool statistics');
      }
      const data = await response.json();
      setStats(data);
      setError(null);
    } catch (err) {
      setError('Error fetching pool statistics. Make sure the pool server is running.');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchStats();
    
    // Auto-refresh every 30 seconds
    const intervalId = setInterval(fetchStats, 30000);
    
    return () => clearInterval(intervalId);
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-foreground mx-auto"></div>
          <p className="mt-4">Loading pool statistics...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen p-8">
        <div className="max-w-4xl mx-auto">
          <header className="mb-8">
            <h1 className="text-3xl font-bold mb-2">Kangaroo Pool</h1>
            <Link href="/" className="text-blue-500 hover:underline">Back to Home</Link>
          </header>
          <div className="bg-red-100 dark:bg-red-900 border border-red-200 dark:border-red-800 rounded-lg p-4 text-red-700 dark:text-red-300">
            {error}
          </div>
          <button 
            onClick={fetchStats}
            className="mt-4 px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen p-8">
      <div className="max-w-6xl mx-auto">
        <header className="mb-8">
          <h1 className="text-3xl font-bold mb-2">Kangaroo Pool</h1>
          <Link href="/" className="text-blue-500 hover:underline">Back to Home</Link>
        </header>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
            <h2 className="text-xl font-semibold mb-2">Total Distinguished Points</h2>
            <p className="text-3xl font-bold">{stats?.totalDP.toLocaleString()}</p>
          </div>
          
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
            <h2 className="text-xl font-semibold mb-2">Active Clients</h2>
            <p className="text-3xl font-bold">{stats?.clients.length || 0}</p>
          </div>
          
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
            <h2 className="text-xl font-semibold mb-2">Last Updated</h2>
            <p className="text-lg">{new Date().toLocaleString()}</p>
            <button 
              onClick={fetchStats} 
              className="mt-2 text-sm text-blue-500 hover:underline"
            >
              Refresh Now
            </button>
          </div>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-lg shadow overflow-hidden">
          <h2 className="text-xl font-semibold p-6 border-b border-gray-200 dark:border-gray-700">
            Client Contributions
          </h2>
          
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead className="bg-gray-50 dark:bg-gray-900">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Bitcoin Address</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Distinguished Points</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Contribution %</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Last Seen</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Client Info</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
                {stats?.clients && stats.clients.length > 0 ? (
                  stats.clients.sort((a, b) => b.dpCount - a.dpCount).map((client, i) => (
                    <tr key={i} className="hover:bg-gray-50 dark:hover:bg-gray-900">
                      <td className="px-6 py-4 whitespace-nowrap text-sm font-medium">
                        {client.address.substring(0, 10)}...{client.address.substring(client.address.length - 5)}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm">
                        {client.dpCount.toLocaleString()}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm">
                        {client.percentage.toFixed(2)}%
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm">
                        {client.lastSeen}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm">
                        {client.clientInfo}
                      </td>
                    </tr>
                  ))
                ) : (
                  <tr>
                    <td colSpan={5} className="px-6 py-4 text-center text-sm text-gray-500 dark:text-gray-400">
                      No clients have connected to the pool yet.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>

        <div className="mt-8 bg-white dark:bg-gray-800 rounded-lg shadow p-6">
          <h2 className="text-xl font-semibold mb-4">How to Connect</h2>
          <div className="bg-gray-100 dark:bg-gray-900 p-4 rounded">
            <code className="text-sm">
              ./kangaroo -c server_ip -p -a YOUR_BITCOIN_ADDRESS [other options]
            </code>
          </div>
          <p className="mt-4 text-sm text-gray-600 dark:text-gray-400">
            Replace YOUR_BITCOIN_ADDRESS with your Bitcoin P2PK address to receive credit for your contributions.
          </p>
          <div className="mt-4">
            <Link 
              href="/pool/client" 
              className="text-blue-500 hover:underline"
            >
              Check your client statistics
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
} 