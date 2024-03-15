import React, { Component } from "react";
import APIServices from '../../../../API/apiservices';
import DataTableComponent from '../../../../Component/DataTableComponent';

const apiServices = new APIServices();

class ODBudget extends Component {

  constructor(props) {
    super(props);
    this.state = {
      ODBudgetData: [],
      ODBudgetColumn: [],
      loading: false
    };
  }

  componentWillReceiveProps = (props) => {
    var self = this;
    const { startDate, endDate, regionId, countryId, cityId } = props;

    // var anciallaryItems = apiServices.getAnciallaryItems();
    // self.setState({ ODBudgetColumn: anciallaryItems[0].columnName, ODBudgetData: anciallaryItems[0].anciallaryItems })
    self.setState({ loading: true, ODBudgetData: [] })
    apiServices.getODBudgetTable(startDate, endDate, regionId, countryId, cityId).then(function (result) {
      self.setState({ loading: false })
      if (result) {
        var columnName = result[0].columnName;
        var ODBudgetData = result[0].rowData;
        self.setState({ ODBudgetData: ODBudgetData, ODBudgetColumn: columnName })
      }
    });

  }

  render() {
    return (
      <div>
          <div className="x_title">
            <h2>OD Budget</h2>
            <ul className="nav navbar-right panel_toolbox">
              {/* <li><a className="collapse-link"><i className="fa fa-chevron-up"></i></a>
                </li>
                <li className="dropdown">
                  <a href="#" className="dropdown-toggle" data-toggle="dropdown" role="button" aria-expanded="false"><i className="fa fa-wrench"></i></a>
                  <ul className="dropdown-menu" role="menu">
                    <li><a href="#">Settings 1</a>
                    </li>
                    <li><a href="#">Settings 2</a>
                    </li>
                  </ul>
                </li> */}
              <li><a href="/topMarkets"><i className="fa fa-line-chart"></i></a></li>
              {/* <li><a className="close-link"><i className="fa fa-close"></i></a></li> */}
            </ul>
          </div>
          <div className="x_content">
            <DataTableComponent
              rowData={this.state.ODBudgetData}
              columnDefs={this.state.ODBudgetColumn}
              dashboard={true}
              loading={this.state.loading}
              height={'22vh'}
            />
          </div>
      </div>
    )
  }
}

export default ODBudget;