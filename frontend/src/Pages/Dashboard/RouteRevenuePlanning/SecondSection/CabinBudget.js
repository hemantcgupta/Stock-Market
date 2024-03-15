import React, { Component } from "react";
import APIServices from '../../../../API/apiservices';
import DataTableComponent from '../../../../Component/DataTableComponent';
import String from '../../../../Constants/validator';
import TotalRow from '../../../../Component/TotalRow';

const apiServices = new APIServices();

class CabinBudget extends Component {

  constructor(props) {
    super(props);
    this.state = {
      cabinData: [],
      cabinColumn: [],
      totalData: [],
      loading: false
    };
  }

  componentWillReceiveProps = (props) => {
    var self = this;
    const { startDate, endDate, routeGroup, regionId, countryId, cityId, routeId, posDashboard } = props;
    self.setState({ loading: true, cabinData: [] })
    const group = String.removeQuotes(routeGroup)
    // this.url = `/rps`;
    this.getRouteURL(group, regionId, countryId, routeId);
    apiServices.getCabinBudget(startDate, endDate, routeGroup, regionId, countryId, cityId, routeId, posDashboard).then(function (result) {
      self.setState({ loading: false })
      if (result) {
        var columnName = result[0].columnName;
        var cabinData = result[0].ancillaryDetails;
        var totalData = result[0].totalData;
        self.setState({ cabinData: cabinData, cabinColumn: columnName, totalData: totalData })
      }
    });

  }

  getRouteURL(routeGroup, regionId, countryId, routeId) {
    let leg = window.localStorage.getItem('LegSelected')
    let flight = window.localStorage.getItem('FlightSelected')

    if (regionId !== 'Null') {
        this.url = `/route?RouteGroup=${routeGroup}&Region=${String.removeQuotes(regionId)}`
    }
    if (countryId !== 'Null') {
        this.url = `/route?RouteGroup=${routeGroup}&Region=${String.removeQuotes(regionId)}&Country=${String.removeQuotes(countryId)}`
    }
    if (routeId !== 'Null') {
        this.url = `/route?RouteGroup=${routeGroup}&Region=${String.removeQuotes(regionId)}&Country=${String.removeQuotes(countryId)}&Route=${String.removeQuotes(routeId)}`
    }
    if (leg !== null && leg !== 'Null' && leg !== '') {
        this.url = `/route?RouteGroup=${routeGroup}&Region=${String.removeQuotes(regionId)}&Country=${String.removeQuotes(countryId)}&Route=${String.removeQuotes(routeId)}&Leg=${String.removeQuotes(leg)}`
    }
    if (flight !== null && flight !== 'Null' && flight !== '') {
        this.url = `/route?RouteGroup=${routeGroup}&Region=${String.removeQuotes(regionId)}&Country=${String.removeQuotes(countryId)}&Route=${String.removeQuotes(routeId)}&Leg=${String.removeQuotes(leg)}&Flight=${String.removeQuotes(flight)}`
    }
}

  render() {
    return (
      <div>
        <div className="x_title">
          <h2>Cabin Budget</h2>
          <ul className="nav navbar-right panel_toolbox">
          <li onClick={() => this.props.history.push(this.url)}><i className="fa fa-line-chart"></i></li>
          </ul>
        </div>
        <div className="x_content">
          {/* <MuiThemeProvider theme={theme}>
              <MUIDataTable
                data={this.state.toptenroutespivottable}
                columns={this.state.toptenroutesperformcolumn}
                options={options}
              />
            </MuiThemeProvider> */}
          <DataTableComponent
            rowData={this.state.cabinData}
            columnDefs={this.state.cabinColumn}
            dashboard={true}
            loading={this.state.loading}
            height={'20vh'}
          />
          <TotalRow
            rowData={this.state.totalData}
            columnDefs={this.state.cabinColumn}
            loading={this.state.loading}
            dashboard={true}
            reducingPadding={true}
          />
        </div>
      </div>
    )
  }
}

export default CabinBudget;