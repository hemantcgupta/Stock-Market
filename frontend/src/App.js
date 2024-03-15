import React from 'react';
import './App.scss';
import './custom.min.scss';
import BaseRouter from './routes';
import { BrowserRouter as Router } from 'react-router-dom';

class App extends React.Component {
  constructor(props) {
    super(props);
  }


  render() {
    return (
      <div className="App">
        <Router>
          <BaseRouter />
        </Router>
      </div>
    );
  }
}

export default App;
