import React, { Component } from "react";
import APIServices from './API/apiservices';
import MUIDataTable from 'mui-datatables';
import { MuiThemeProvider } from '@material-ui/core/styles';
import theme from './pivottableStyle.js';

const apiServices = new APIServices();

class PivotTable extends Component {

  constructor(props) {
    super(props);
    this.state = { 
        channelwiseperformpivottable: [],
        channelwiseperformcolumn: [],
        toptenroutespivottable: [],
        toptenroutesperformcolumn: [],
    };
  }

  componentDidMount() {
    var self = this;
    apiServices.getChannelWisePerformPivot().then(function (result) {
        var columnName = result[0].columnName;
        var channelwiseperformpivot = result[0].channelwiseperformtableDatas;
        self.setState({ channelwiseperformpivottable: channelwiseperformpivot, channelwiseperformcolumn: columnName})
    });

    apiServices.gettop_10_routesPivot().then(function (result) {
        var columnName = result[0].columnName;
        var toptenroutespivot = result[0].toptenroutespivotDatas;
        self.setState({ toptenroutespivottable: toptenroutespivot, toptenroutesperformcolumn: columnName})
    });

  }

  render() {

    const options = {
        responsive: 'scroll',
        filter: false,
        download:true,
        print:false,
        selectableRows: 'none',
        filterType: 'dropdown',
        rowsPerPage: 6,
        rowsPerPageOptions: [6, 25, 50],
    };
      
    return (
      <div>

        <div className="col-md-4 col-sm-4 col-xs-12">
          <div className="x_panel">
            <div className="x_title">
              <h2>Channel Wise Performance</h2>
              {/* <ul className="nav navbar-right panel_toolbox">
                <li><a className="collapse-link"><i className="fa fa-chevron-up"></i></a>
                </li>
                <li className="dropdown">
                  <a href="#" className="dropdown-toggle" data-toggle="dropdown" role="button" aria-expanded="false"><i className="fa fa-wrench"></i></a>
                  <ul className="dropdown-menu" role="menu">
                    <li><a href="#">Settings 1</a>
                    </li>
                    <li><a href="#">Settings 2</a>
                    </li>
                  </ul>
                </li>
                <li><a className="close-link"><i className="fa fa-close"></i></a>
                </li>
              </ul> */}
              <div className="clearfix"></div>
            </div>
            <div className="x_content">
                <MuiThemeProvider theme={theme}>
                    <MUIDataTable 
                      data={this.state.channelwiseperformpivottable} 
                      columns={this.state.channelwiseperformcolumn} 
                      options={options} 
                    />
                </MuiThemeProvider>
            </div>
          </div>
        </div>

        <div className="col-md-4 col-sm-4 col-xs-12">
          <div className="x_panel">
            <div className="x_title">
              <h2>Top 10 Routes</h2>
              {/* <ul className="nav navbar-right panel_toolbox">
                <li><a className="collapse-link"><i className="fa fa-chevron-up"></i></a>
                </li>
                <li className="dropdown">
                  <a href="#" className="dropdown-toggle" data-toggle="dropdown" role="button" aria-expanded="false"><i className="fa fa-wrench"></i></a>
                  <ul className="dropdown-menu" role="menu">
                    <li><a href="#">Settings 1</a>
                    </li>
                    <li><a href="#">Settings 2</a>
                    </li>
                  </ul>
                </li>
                <li><a className="close-link"><i className="fa fa-close"></i></a>
                </li>
              </ul> */}
              <div className="clearfix"></div>
            </div>
            <div className="x_content">
                <MuiThemeProvider theme={theme}>
                  <MUIDataTable 
                    data={this.state.toptenroutespivottable} 
                    columns={this.state.toptenroutesperformcolumn} 
                    options={options} 
                  />
                </MuiThemeProvider>
            </div>
          </div>
        </div>

    </div>
    )
  }
}

export default PivotTable;