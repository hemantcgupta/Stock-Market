import React, { Component } from 'react';
import APIServices from './API/apiservices';
import ChartModelDetails from './Component/chartModel'
import MUIDataTable from 'mui-datatables';
import { MuiThemeProvider } from '@material-ui/core/styles';
import theme from './postableStyle.js';
import $ from 'jquery';
import './App.css';
// import './breadcrumb.css';

const apiServices = new APIServices();
var bcData = [];
var count = 1;
var inputLevel;
var gettingRegion;

class POSDetail extends Component {

  constructor(props) {
    super(props);
    this.state = { 
        posMonthDetails: [],
        monthcolumns:[],
        regioncolumn: [],
        posRegionDatas: [],
        modelRegionDatas:[],
        modelregioncolumn:[],
        tableDatas: true,
        gettingMonth: 'Null',
        tableTitle: '',
        monthTableTitle: '',
        tabLevel:'',
    };
  }

  componentDidMount() {
    var self = this;
    apiServices.getPOSMonthTables().then(function (result) {
        var columnName = result[0].columnName;
        var posMonthdata = result[0].posmonthtableDatas;
        var tableTitle = result[0].monthTableTitle;
        self.setState({ posMonthDetails: posMonthdata, monthcolumns:columnName, monthTableTitle: tableTitle})
    });

    apiServices.getPOSRegionTables(this.state.gettingMonth).then(function (result) {
        var columnName = result[0].columnName;
        var posRegionata = result[0].posregiontableDatas;
        var tableTitle = result[0].tableTitle;
        self.setState({ posRegionDatas: posRegionata, regioncolumn: columnName, tableTitle: tableTitle})
    });
  }

  handleClick(rowData) {
      var monththis  = this;
      var gettingMonth = window.monthNameToNum(rowData[0]);
      var getColName = $('.regionTable thead tr th:nth-child(1)').text();

      console.log("Test getColName Value : ",getColName );
      console.log("Test gettingRegion Value : ",gettingRegion );
      if(getColName === 'RegionName'){
        apiServices.getPOSRegionTables(gettingMonth).then(function (result) {
          inputLevel = "R";
          var columnName = result[0].columnName;
          var posRegionata = result[0].posregiontableDatas;
          var tableTitle = result[0].tableTitle;
          monththis.setState({ posRegionDatas: posRegionata, regioncolumn:columnName, gettingMonth: gettingMonth, tableTitle: tableTitle})
        });

      }else if(getColName === 'CountryName'){
        inputLevel = "C";
        apiServices.getPOSCountryDetails(gettingMonth, gettingRegion).then(function (result) {
          var columnName = result[0].columnName;        
          var posDatas = result[0].posCountrytableDatas;
          var tableTitle = result[0].tableTitle;
          monththis.setState({ posRegionDatas: posDatas, regioncolumn:columnName,gettingMonth: gettingMonth, tableTitle: tableTitle})
        });
      }else if(getColName === 'POS'){
        inputLevel = "P";
        apiServices.getPOSDetails(gettingMonth, gettingRegion).then(function (result) {
          var columnName = result[0].columnName;       
          var posDatas = result[0].posregiontableDatas;
          var tableTitle = result[0].tableTitle;
          monththis.setState({ posRegionDatas: posDatas, regioncolumn:columnName ,gettingMonth: gettingMonth , tableTitle: tableTitle})
        });

      }else{
        inputLevel = "O";
        apiServices.getPOSODDetails(gettingMonth, gettingRegion).then(function (result) {
          var columnName = result[0].columnName;       
          var posDatas = result[0].posregiontableDatas;
          var tableTitle = result[0].tableTitle;
          monththis.setState({ posRegionDatas: posDatas, regioncolumn:columnName, gettingMonth: gettingMonth, tableTitle: tableTitle})
        });  

      }
      bcData = [];
    // if(count === 1){
    //   bcData.push({"key":"M","val":rowData[0],"title":"Month"});
    // } else {
    //   bcData = [];
    //   bcData.push({"key":"M","val":rowData[0],"title":"Month"});
    // }
    // count += 1;
  }
  
  RegionhandleClick(rowData) {
    var regionthis  = this;
    gettingRegion = rowData[0];
    var gettingMonth = this.state.gettingMonth;
    var getColName = $('.regionTable thead tr th:nth-child(1)').text();

    // var inputLevel;

    if(getColName === 'RegionName'){
      inputLevel = "C";
      console.log("gettingMonth : ",gettingMonth);
      apiServices.getPOSCountryDetails(gettingMonth, gettingRegion).then(function (result) {
        var columnName = result[0].columnName;        
        var posDatas = result[0].posCountrytableDatas;
        var tableTitle = result[0].tableTitle;

        // tabVal = "C";
        bcData.push({"key":inputLevel,"val":gettingRegion,"title":"Region"});
        console.log("Demo Data1 : ", bcData);

        regionthis.setState({ posRegionDatas: posDatas, regioncolumn:columnName, tableTitle: tableTitle, tabLevel:inputLevel})
      });

      apiServices.getMonthWise(gettingRegion, inputLevel).then(function (result) {
        var columnName = result[0].columnName;
        var posRegionata = result[0].posmonthtableDatas;
        var tableTitle = result[0].monthTableTitle;
        regionthis.setState({ posMonthDetails: posRegionata, monthcolumns:columnName, monthTableTitle: tableTitle})
      });
      // bcData.push(rowData[0]);
    } else if(getColName === 'CountryName') {
      inputLevel = "P";
      console.log("gettingMonth : ",gettingMonth);
      apiServices.getPOSDetails(gettingMonth, gettingRegion).then(function (result) {
        var columnName = result[0].columnName;       
        var posDatas = result[0].posregiontableDatas;
        var tableTitle = result[0].tableTitle;
       
        bcData.push({"key":inputLevel,"val":gettingRegion,"title":"Country"});
        console.log("Demo Data 2: ", bcData);

        regionthis.setState({ posRegionDatas: posDatas, regioncolumn:columnName , tableTitle: tableTitle})
      });
      apiServices.getMonthWise(gettingRegion, inputLevel).then(function (result) {
        var columnName = result[0].columnName;
        var posRegionata = result[0].posmonthtableDatas;
        var tableTitle = result[0].monthTableTitle;
        regionthis.setState({ posMonthDetails: posRegionata, monthcolumns:columnName, monthTableTitle: tableTitle})
      });
  
      // bcData.push(rowData[0]);
    } else if(getColName === 'POS') {
      inputLevel = "O";
      console.log("gettingMonth : ",gettingMonth);
      apiServices.getPOSODDetails(gettingMonth, gettingRegion).then(function (result) {
        var columnName = result[0].columnName;       
        var posDatas = result[0].posregiontableDatas;
      var tableTitle = result[0].tableTitle;

        // tabVal = "O";
        bcData.push({"key":inputLevel,"val":gettingRegion,"title":"POS"});
        console.log("Demo Data3 : ", bcData);

        regionthis.setState({ posRegionDatas: posDatas, regioncolumn:columnName, tableTitle: tableTitle})
      });  
      apiServices.getMonthWise(gettingRegion, inputLevel).then(function (result) {
        var columnName = result[0].columnName;
        var posRegionata = result[0].posmonthtableDatas;
        var tableTitle = result[0].monthTableTitle;
        regionthis.setState({ posMonthDetails: posRegionata, monthcolumns:columnName, monthTableTitle: tableTitle})
      });
  

    }
    count = 2;
  }

  posHandleClick = (e) => {
    var region  = this;
    console.log("Pos Handle Click Function is Triggering.....");
    apiServices.getPOSRegionTables(this.state.gettingMonth).then(function (result) {
      var columnName = result[0].columnName;
      var posRegionata = result[0].posregiontableDatas;
      var tableTitle = result[0].tableTitle;
      region.setState({ posRegionDatas: posRegionata, regioncolumn: columnName, tableTitle: tableTitle})
    });
  }

  odHandleClick = (e) => {
    console.log("O & D Handle Click Function is Triggering.....");
    var self = this;
    apiServices.getPOSMonthTables().then(function (result) {
        var columnName = result[0].columnName;
        var posMonthdata = result[0].posmonthtableDatas;
        var tableTitle = result[0].monthTableTitle;
        self.setState({ posRegionDatas: posMonthdata, regioncolumn:columnName, tableTitle: tableTitle})
    });
  }

  agentHandleClick = (e) => {
    var self = this;
    // console.log("Agent Handle Click Function is Triggering Level is.....", self.state.tabLevel);
    apiServices.getAgentDetails().then(function (result) {
        var columnName = result[0].columnName;
        var posAgentData = result[0].posregiontableDatas;
        var tableTitle = result[0].tableTitle;
        self.setState({ posRegionDatas: posAgentData, regioncolumn:columnName, tableTitle: tableTitle})
    });
  }

  homeHandleClick = (e) =>{
    var self = this;
    apiServices.getPOSMonthTables().then(function (result) {
        var columnName = result[0].columnName;
        var posMonthdata = result[0].posmonthtableDatas;
        var tableTitle = result[0].monthTableTitle;
        self.setState({ posMonthDetails: posMonthdata, monthcolumns:columnName, monthTableTitle: tableTitle})
    });

    apiServices.getPOSRegionTables(this.state.gettingMonth).then(function (result) {
        var columnName = result[0].columnName;
        var posRegionata = result[0].posregiontableDatas;
        var tableTitle = result[0].tableTitle;
        self.setState({ posRegionDatas: posRegionata, regioncolumn: columnName, tableTitle: tableTitle})
    });
    bcData = [];
  }

  listHandleClick = (e) => {
    var regionthis  = this;
    gettingRegion = e.target.id;
    var gettingMonth = this.state.gettingMonth;
    var getColName = e.target.title;
    
    // var inputLevel;

    var indexEnd = bcData.findIndex(function(e) {
      return e.val == gettingRegion;
    })
    var removeArrayIndex = bcData.slice(0,indexEnd+1);
    bcData = removeArrayIndex;

    if(getColName==='Region'){
      inputLevel = "C";
      apiServices.getPOSCountryDetails(gettingMonth, gettingRegion).then(function (result) {
        var columnName = result[0].columnName;        
        var posDatas = result[0].posCountrytableDatas;
        var tableTitle = result[0].tableTitle;
        regionthis.setState({ posRegionDatas: posDatas, regioncolumn:columnName, tableTitle: tableTitle})
      });
      apiServices.getMonthWise(gettingRegion, inputLevel).then(function (result) {
        var columnName = result[0].columnName;
        var posRegionata = result[0].posmonthtableDatas;
        var tableTitle = result[0].monthTableTitle;
        regionthis.setState({ posMonthDetails: posRegionata, monthcolumns:columnName, monthTableTitle: tableTitle})
      });
    } else if(getColName ==='Country') {
      inputLevel = "P";
      apiServices.getPOSDetails(gettingMonth, gettingRegion).then(function (result) {
        var columnName = result[0].columnName;       
        var posDatas = result[0].posregiontableDatas;
        var tableTitle = result[0].tableTitle;
        regionthis.setState({ posRegionDatas: posDatas, regioncolumn:columnName , tableTitle: tableTitle})
      });
      apiServices.getMonthWise(gettingRegion, inputLevel).then(function (result) {
        var columnName = result[0].columnName;
        var posRegionata = result[0].posmonthtableDatas;
        var tableTitle = result[0].monthTableTitle;
        regionthis.setState({ posMonthDetails: posRegionata, monthcolumns:columnName, monthTableTitle: tableTitle})
      });
    } else if(getColName ==='POS') {
      inputLevel = "O";
      apiServices.getPOSODDetails(gettingMonth, gettingRegion).then(function (result) {
        var columnName = result[0].columnName;       
        var posDatas = result[0].posregiontableDatas;
        var tableTitle = result[0].tableTitle;
        regionthis.setState({ posRegionDatas: posDatas, regioncolumn:columnName, tableTitle: tableTitle})
      });
      apiServices.getMonthWise(gettingRegion, inputLevel).then(function (result) {
        var columnName = result[0].columnName;
        var posRegionata = result[0].posmonthtableDatas;
        var tableTitle = result[0].monthTableTitle;
        regionthis.setState({ posMonthDetails: posRegionata, monthcolumns:columnName, monthTableTitle: tableTitle})
      });
    }

  }
  cellhandleClick = (cellIndex,rowData) => {
    console.log("row cellIndex : ", cellIndex);
    console.log("row Data : ", rowData);
    if(rowData.colIndex === 1){
      rowData.event.stopPropagation();
      console.log("Action is Triggering...............");
      var self = this;
      apiServices.getPOSRegionTables(this.state.gettingMonth).then(function (result) {
        var columnName = result[0].columnName;
        var posRegionata = result[0].posregiontableDatas;
        self.setState({ modelRegionDatas: posRegionata, modelregioncolumn: columnName})
      });
    }
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
      onRowClick: (rowData, rowIndex) => {
        var rowindex = (rowIndex.rowIndex) + 1;
        $(".monthTable tbody tr").css("background-color","#fff");
        $(".monthTable tbody tr:nth-child("+ rowindex +")").css("color", "rgba(255, 255, 255, 0.87)").css("background-color", "#537790c2");   
        this.handleClick(rowData);
      },
      onCellClick: (cellIndex,rowData) => {this.cellhandleClick(cellIndex,rowData)},
    };

    const regionOptions = {
      responsive: 'scroll',
      filter: false,
      download:true,
      print:false,
      selectableRows: 'none',
      filterType: 'dropdown',
      rowsPerPage: 6,
      rowsPerPageOptions: [6, 25, 50],
      onRowClick: rowData => {this.RegionhandleClick(rowData)},
      onCellClick: (cellIndex,rowData) => {this.cellhandleClick(cellIndex,rowData)},
    };

    console.log("Check Render Function ", bcData);

    return (
      <div>
        <div className="row">
          <div className="col-md-12 col-sm-12 col-xs-12">
            <section>
              <nav>
                <ol className="cd-breadcrumb">
                <li onClick={this.homeHandleClick} > Home </li>
                {bcData.map((item) =>
                  <li onClick={this.listHandleClick} data-value={item.key}  id={item.val} title={item.title}> {item.val} </li>
                )}
                </ol>
              </nav>
            </section>

            <div className="x_panel">
              <div className="x_content">
                <MuiThemeProvider theme={theme}>
                  <MUIDataTable className="monthTable"
                    data={this.state.posMonthDetails} 
                    columns={this.state.monthcolumns} 
                    options={options} 
                    title={this.state.monthTableTitle} 
                />
                </MuiThemeProvider>
              </div>
            </div>
          </div>
        </div>

        <div className="row">
          <div className="col-md-12 col-sm-12 col-xs-12">
            <div className="x_panel">

            <div className="btn-group">
              <button type="button" onClick = {this.posHandleClick}  className="btn btn-primary button-1" style={{ "margin-left": "10px" }}>POS Flow</button>
              <button type="button" onClick = {this.odHandleClick}  className="btn btn-primary button-1" style={{ "margin-left": "10px" }}>O&D Flow</button>
              {/ <button type="button" onClick = {this.agentHandleClick()}  className="btn btn-primary button-1" style={{ "margin-left": "10px" }}>Agency Flow</button> /}
            </div>

              <div className="x_content">
                <MuiThemeProvider theme={theme}>
                  <MUIDataTable className = 'regionTable'
                    data={this.state.posRegionDatas} 
                    columns={this.state.regioncolumn}
                    options={regionOptions}
                    title={this.state.tableTitle} 
                  />
                </MuiThemeProvider>
              </div>
            </div>
          </div>
        </div>

        <div> 
          <ChartModelDetails datas = {this.state.modelRegionDatas} columns = {this.state.modelregioncolumn}/>
        </div>

      </div>

    );
  }
}

export default POSDetail;