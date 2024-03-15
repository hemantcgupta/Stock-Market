import React, { Component } from "react";
import APIServices from '../../../../API/apiservices';
import DataTableComponent from '../../../../Component/DataTableComponent'
import String from '../../../../Constants/validator';

const apiServices = new APIServices();

class TopBottomRoutesRP extends Component {

  constructor(props) {
    super(props);
    this.state = {
      routesData: [],
      routesColumn: [],
      headerName: '',
      loading: false
    };
  }

  componentWillReceiveProps = (props) => {
    this.getRouteData(props);
  }


  getRouteData(props) {
    var self = this;
    const { startDate, endDate, routeGroup, regionId, countryId, routeId, typeofCost } = props;
    const group = String.removeQuotes(routeGroup)
    this.url = `/routeProfitabilitySolution?RouteGroup=${group}`;
    this.getRouteURL(group, regionId, countryId, routeId, typeofCost);
    self.setState({ loading: true, routesData: [] })
    apiServices.getTopTenRouteBudgetTable(startDate, endDate, routeGroup, regionId, countryId, routeId, typeofCost).then(function (result) {
      self.setState({ loading: false })
      if (result) {
        var columnName = result[0].columnName;
        var rowData = result[0].rowData;
        self.setState({ routesData: rowData, routesColumn: columnName })
      }
    });
    // }
  }

  // headerChange = (e) => {
  //   this.setState({ headerName: e.target.value }, () => this.getRouteData(this.props))
  // }

  getRouteURL(routeGroup, regionId, countryId, routeId, typeofCost) {
    let leg = window.localStorage.getItem('LegSelected')
    let flight = window.localStorage.getItem('FlightSelected')

    if (regionId !== 'Null') {
      this.url = `/routeProfitabilitySolution?RouteGroup=${routeGroup}&Region=${String.removeQuotes(regionId)}`
    }
    if (countryId !== 'Null') {
      this.url = `/routeProfitabilitySolution?RouteGroup=${routeGroup}&Region=${String.removeQuotes(regionId)}&Country=${String.removeQuotes(countryId)}`
    }
    if (routeId !== 'Null') {
      this.url = `/routeProfitabilitySolution?RouteGroup=${routeGroup}&Region=${String.removeQuotes(regionId)}&Country=${String.removeQuotes(countryId)}&Route=${String.removeQuotes(routeId)}`
    }
    if (leg !== null && leg !== 'Null' && leg !== '') {
      this.url = `/routeProfitabilitySolution?RouteGroup=${routeGroup}&Region=${String.removeQuotes(regionId)}&Country=${String.removeQuotes(countryId)}&Route=${String.removeQuotes(routeId)}&Leg=${String.removeQuotes(leg)}`
    }
    if (flight !== null && flight !== 'Null' && flight !== '') {
      this.url = `/routeProfitabilitySolution?RouteGroup=${routeGroup}&Region=${String.removeQuotes(regionId)}&Country=${String.removeQuotes(countryId)}&Route=${String.removeQuotes(routeId)}&Leg=${String.removeQuotes(leg)}&Flight=${String.removeQuotes(flight)}`
    }
  }

  render() {

    return (
      <div>
        <div className="x_title">
          {/* <select class="select header-dropdown" onChange={(e) => this.headerChange(e)}>
            <option>TOP 10 ROUTES</option>
            <option>BOTTOM 5 ROUTES</option>
          </select> */}
          <h2>TOP 10 ROUTES</h2>
          <ul className="nav navbar-right panel_toolbox">
            <li onClick={() => this.props.history.push(`${this.url}&route=${true}`)}><i className="fa fa-line-chart"></i></li>
          </ul>
        </div>
        <DataTableComponent
          rowData={this.state.routesData}
          columnDefs={this.state.routesColumn}
          dashboard={true}
          routeDashboard={true}
          loading={this.state.loading}
          height={'22vh'}
        />
      </div>
    )
  }
}

export default TopBottomRoutesRP;