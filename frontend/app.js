import React, { useState } from 'react';
import './App.css';
import 'bootstrap/dist/css/bootstrap.min.css';
// import { connect } from 'react-redux';


const App = (props) => {

  const mydata = [
    {
      "id": "1",
      "sku": "Tiger Nixon",
      "sale_price": "$320,800",
      "units": "5421"
    },
    {
      "id": "2",
      "sku": "Tiger Nixon",
      "sale_price": "$320,800",
      "units": "5421"
    },
    {
      "id": "3",
      "sku": "Tiger Nixon",
      "sale_price": "$320,800",
      "units": "5421"
    },
    {
      "id": "4",
      "sku": "Tiger Nixon",
      "sale_price": "$320,800",
      "units": "5421"
    }

  ]
  const [tbl_data, settbl_data] = useState(mydata);

  
  // FOR BACKEND CALL UNCOMMENT BELOW CODE

  // useEffect(() => {

  //   // FLASK API ENDPOINT
  //   fetch(`http://localhost:8000/url/`, {
  //     method: 'POST',
  //     headers: { 'Content-Type': 'application/json' },
  //     body: JSON.stringify({ sku_id: 123456 })
  //   })
  //     .then(res => res.json())
  //     .then(json => settbl_data(json));

  // }, []);


  return (
    <>

      <div className="container">
        <h2 style={{ textAlign: "center" }}>Sales History</h2>
        <table className="table">
          <thead>
            <tr>
              <th>SKU ID</th>
              <th>SKU</th>
              <th>Sale Price</th>
              <th>Unit Sold</th>
            </tr>
          </thead>
          <tbody>
            {
              tbl_data.map((item) => (
                <tr key={item.id}>
                  <td>{item.id}</td>
                  <td>{item.sku}</td>
                  <td>{item.sale_price}</td>
                  <td>{item.units}</td>
                </tr>
              ))
            }
          </tbody>
        </table>
      </div>

    </>
  );
}


export default App;