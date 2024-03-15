import React, { Component } from "react";
import APIServices from '../../../../API/apiservices';
import DataTableComponent from '../../../../Component/DataTableComponent'
import String from '../../../../Constants/validator';

const apiServices = new APIServices();

class TopBottomRoutesBudget extends Component {

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
    const { startDate, endDate, routeGroup, regionId, countryId, routeId } = props;
    const group = String.removeQuotes(routeGroup)
    // this.url = `/rps`;
    this.getRouteURL(group, regionId, countryId, routeId);
    if (this.state.headerName === 'BOTTOM 5 ROUTES') {
      self.setState({ loading: true, routesData: [] })
      apiServices.getRouteBudgetTable(startDate, endDate, routeGroup, regionId, countryId, routeId, 'asc').then(function (result) {
        self.setState({ loading: false })
        if (result) {
          var columnName = result[0].columnName;
          var rowData = result[0].rowData;
          self.setState({ routesData: rowData, routesColumn: columnName })
        }
      });
    } else {
      self.setState({ loading: true, routesData: [] })
      apiServices.getRouteBudgetTable(startDate, endDate, routeGroup, regionId, countryId, routeId, 'desc').then(function (result) {
        self.setState({ loading: false })
        if (result) {
          var columnName = result[0].columnName;
          var rowData = result[0].rowData;
          self.setState({ routesData: rowData, routesColumn: columnName })
        }
      });
    }
  }

  headerChange = (e) => {
    this.setState({ headerName: e.target.value }, () => this.getRouteData(this.props))
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
          <select class="select header-dropdown" onChange={(e) => this.headerChange(e)}>
            <option>TOP 5 ROUTES</option>
            <option>BOTTOM 5 ROUTES</option>
          </select>
          <ul className="nav navbar-right panel_toolbox">
            <li onClick={() => this.props.history.push(this.url)}><i className="fa fa-line-chart"></i></li>
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

export default TopBottomRoutesBudget;