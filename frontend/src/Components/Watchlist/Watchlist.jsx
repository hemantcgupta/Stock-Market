// Watchlist.jsx
import React from "react";
import { ChevronDown, PlusCircle, MoreHorizontal } from "lucide-react";

const Watchlist = () => {
  const watchlistData = [
    { symbol: "SPX", last: 5375.86, chg: 88.1, chgPercent: 1.67 },
    { symbol: "SENSEX", last: 79801.43, chg: -315.06, chgPercent: -0.39 },
    { symbol: "NIFTY", last: 24246.7, chg: -82.25, chgPercent: -0.34 },
    { symbol: "CNXIT", last: 35307.1, chg: -107.55, chgPercent: -0.3 },
    { symbol: "BANKNIFT", last: 55201.4, chg: -168.65, chgPercent: -0.3 },
  ];

  return (
    <div className="watchlist-container">
      <div className="header">
        <button className="dropdown">
          Watchlist<ChevronDown className="icon" /> 
        </button>
      </div>

      <div className="table-container">
        <table>
          <thead>
            <tr>
              <th>Symbol</th>
              <th>Last</th>
              <th>Chg</th>
              <th>Chg%</th>
            </tr>
          </thead>
          <tbody>
            {watchlistData.map((item, index) => (
              <tr key={index} className={item.chg > 0 ? "positive" : "negative"}>
                <td>
                  <span className={`status-indicator ${item.chg > 0 ? "positive" : "negative"}`} />
                  {item.symbol}
                </td>
                <td>{item.last.toFixed(2)}</td>
                <td>{item.chg.toFixed(2)}</td>
                <td>{item.chgPercent.toFixed(2)}%</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default Watchlist;